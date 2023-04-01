﻿// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                           Copyright (c) 2021 ILGPU Project
//                                    www.ilgpu.net
//
// File: Program.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;
using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace MatrixMultiplySparse
{
    class Program
    {
        /// <summary>
        /// Main entry point
        /// </summary>
        static void Main()
        {
            // Performs a sanity check on the matrix multiply implementations against known inputs and output.
            var sanityMatrixA = new float[4, 3]
            {
                {  1,   2,   3 },
                {  4,   5,   6 },
                {  7,   8,   9 },
                { 10,  11,  12 },
            };

            var sanityMatrixB = new float[3, 5]
            {
                { 13, 14, 15, 16, 17 },
                { 18, 19, 20, 21, 22 },
                { 23, 24, 25, 26, 27 },
            };

            var sanityMatrixP = new float[4, 3]
            {
                {1, 1, 0},
                {1, 1, 0},
                {0, 0, 0},
                {0, 0, 0},
            };

            var sanityResult = new float[4, 5]
            {
                {  49,      52,      55,      58,      61},
                { 142,     151,     160,     169,     178},
                {   0,       0,       0,       0,       0},
                {   0,       0,       0,       0,       0},
            };

            var sanityMatrixBt = Utils.MatrixTranspose(sanityMatrixB);

            var naiveResult = PABt(sanityMatrixP, sanityMatrixA, sanityMatrixBt);
            Console.WriteLine("Naive implementation result");
            Utils.PrintMatrix(naiveResult);
            Debug.Assert(Utils.MatrixEqual(naiveResult, sanityResult));

            var simpleResult = PABt_sparse_simple(sanityMatrixP, sanityMatrixA, sanityMatrixBt);
            Debug.Assert(Utils.MatrixEqual(simpleResult, sanityResult));
            Console.WriteLine();
            Console.WriteLine("----------------------------------------------------------------------------");
            Console.WriteLine("Naive sparse result");
            Utils.PrintMatrix(simpleResult);

            var efficientResult = PABt_sparse_efficient(sanityMatrixP, sanityMatrixA, sanityMatrixBt);
            Debug.Assert(Utils.MatrixEqual(simpleResult, sanityResult));
            Console.WriteLine();
            Console.WriteLine("----------------------------------------------------------------------------");
            Console.WriteLine("Efficient sparse result");
            Utils.PrintMatrix(efficientResult);

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            // Accelerated implementations

            var sanityMatrixSBt = SparseMatrix.Create(sanityMatrixBt);
            using var context = Context.CreateDefault();
            foreach (var device in context)
            {
                using var accelerator = device.CreateAccelerator(context);
                var acceleratedResult = MatrixMultiplyAcceleratedSparse(accelerator, sanityMatrixP, sanityMatrixA, sanityMatrixSBt);
                Debug.Assert(Utils.MatrixEqual(acceleratedResult, sanityResult));

                Console.WriteLine("----------------------------------------------------------------------------");
                Console.WriteLine($"- Accelerated sparse implementation on {accelerator}");
                Utils.PrintMatrix(acceleratedResult);
            }

            // Prepare random matrices
            /*
            const int m = 500;
            const int n = 500;
            const int k = 500;

            var aMatrix = CreateRandomMatrix(m, k);
            var bMatrix = CreateRandomMatrix(k, n);
            var cMatrix = MatrixMultiplyNaive(aMatrix, bMatrix);

            RunMatrixMultiply(aMatrix, bMatrix, cMatrix);
            */
        }

        #region Helper functions

        /// <summary>
        /// Compute (P&&A) * B'
        /// </summary>
        static float[,] PABt(float[,] P, float[,] A, float[,] B)
        {
            var Bt = Utils.MatrixTranspose(B);
            var P_and_A = Utils.MatrixMask(A, P);
            var result = Utils.MatrixMultiplyNaive(P_and_A, Bt);
            return result;
        }


        /// <summary>
        /// Compute (P&&A) * B'
        /// Using Sparse Matrix routines
        /// </summary>
        static float[,] PABt_sparse_simple(float[,] P, float[,] A, float[,] B)
        {
            var Bt = Utils.MatrixTranspose(B);
            var SBt = SparseMatrix.Create(Bt);
            var result = PABt_mult_simple(P, A, SBt);
            return result;
        }


        // Compute (P&&A) * SBt  where SBt = sparse matrix
        static float[,] PABt_mult_simple(float[,] P, float[,] A, SparseMatrix SBt) {

            var m = A.GetLength(0);
            var ka = A.GetLength(1);
            var kb = SBt.NumRows;
            var n = SBt.NumColumns;

            if (ka != kb)
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(A));

            var c = new float[m, n];
            for (var x = 0; x < m; x++)
            {
                for (var y = 0; y < n; y++)
                {
                    c[x, y] = 0;
                    for (var z = 0; z < ka; z++)
                        c[x, y] += P[x,z] * A[x, z] * SBt[z, y];
                }
            }

            return c;
        }


        // Compute (P&&A) * B'  where B = dense matrix
        static float[,] PABt_sparse_efficient(float[,] P, float[,] A, float[,] B)
        {
            var SB = SparseMatrix.Create(B);
            var result = PABt_mult_efficient(P, A, SB);
            return result;
        }


        // Compute (P&&A) * SB'  where SB = sparse matrix
        // This computes an implicit transpose for SB
        // Uses dense matrix information to only multiply nonzero elements
        static float[,] PABt_mult_efficient(float[,] P, float[,] A, SparseMatrix SB) {

            var m = A.GetLength(0);
            var ka = A.GetLength(1);
            var kb = SB.NumColumns;
            var n = SB.NumRows ;

            int[,] neighbors = SB.neighbors;
            int[] numNeighbors = SB.numNeighbors;
            float[,] edgeWeights = SB.edgeWeights;

            if (ka != kb)
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(A));

            var c = new float[m, n];

            for (var x = 0; x < m; x++)
            {
                for (var y = 0; y < n; y++)
                {
                    float dot = 0;
                    int nonZero = numNeighbors[y];  // implicit transpose, array stored row-wise
                    for (var z = 0; z < nonZero; z++) {
                        int colIdx = neighbors[y, z];
                        dot += P[x, colIdx] * A[x, colIdx] * edgeWeights[y, z];
                    }
                    c[x, y] = dot;
                }
            }

            return c;
        }

        #endregion

        
        #region Accelerated algorithm

        /// <summary>
        /// Compute (P&&A) * SB'  where SB = sparse matrix.
        /// </summary>
        /// <param name="accelerator">The Accelerator to run the multiplication on</param>
        /// <param name="P">A dense MxK MASK matrix</param>
        /// <param name="A">A dense MxK matrix</param>
        /// <param name="SB">A sparse NxK matrix</param>
        /// <returns>A dense MxN matrix</returns>
        static float[,] MatrixMultiplyAcceleratedSparse(Accelerator accelerator, float[,] P, float[,] A,  SparseMatrix SB)
        {
            var m = A.GetLength(0);
            var ka = A.GetLength(1);
            var kb = SB.NumColumns;
            var n = SB.NumRows;

            int[,] neighbors = SB.neighbors;
            int[] numNeighbors = SB.numNeighbors;
            float[,] edgeWeights = SB.edgeWeights;


            if (ka != kb)
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(A));

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<int, Stride2D.DenseX>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMultiplyAcceleratedKernelSparse);

            using var pBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            using var aBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            using var outBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, n));

            using var neighborsBuffer = accelerator.Allocate2DDenseX<int>(new Index2D(neighbors.GetLength(0), neighbors.GetLength(1)));
            using var numNeighborsBuffer = accelerator.Allocate1D<int>(neighbors.GetLength(0));
            using var edgeWeightsBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(edgeWeights.GetLength(0), edgeWeights.GetLength(1)));
            
            
            pBuffer.CopyFromCPU(P);
            aBuffer.CopyFromCPU(A);
            
            // Sparse matrix SB members
            neighborsBuffer.CopyFromCPU(neighbors);
            numNeighborsBuffer.CopyFromCPU(numNeighbors);
            edgeWeightsBuffer.CopyFromCPU(edgeWeights);

            kernel(outBuffer.Extent.ToIntIndex(), pBuffer.View, aBuffer.View, 
                neighborsBuffer.View, numNeighborsBuffer.View, edgeWeightsBuffer.View,
                outBuffer.View);

            // Reads data from the GPU buffer into a new CPU array.
            // Implicitly calls accelerator.DefaultStream.Synchronize() to ensure
            // that the kernel and memory copy are completed first.
            return outBuffer.GetAsArray2D();
        }


        /// <summary>
        /// The matrix multiplication kernel that runs on the accelerated device.
        /// </summary>
        /// <param name="index">Current matrix index</param>
        /// <param name="pView">An input matrix MASK of size MxK</param>
        /// <param name="aView">An input matrix of size MxK</param>
        /// <param name="neighborsView"> B.neighbors member: A sparse matrix B of size NxK (will transpose)</param>
        /// <param name="numNeighborsView"> B.numNeighborsView member: A sparse matrix B of size NxK (will transpose)</param>
        /// <param name="edgeWeightsView"> B.edgeWeightsView member: A sparse matrix B of size NxK (will transpose)</param>
        /// <param name="cView">An output matrix of size MxN</param>
        /// Uses dense matrix information to only multiply nonzero elements
        static void MatrixMultiplyAcceleratedKernelSparse(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> pView,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<int, Stride2D.DenseX> neighborsView,
            ArrayView1D<int, Stride1D.Dense> numNeighborsView,
            ArrayView2D<float, Stride2D.DenseX> edgeWeightsView,
            ArrayView2D<float, Stride2D.DenseX> outView)
        {
            var x = index.X;
            var y = index.Y;
            float dot = 0.0f;

            // implicit transpose of B, array stored row-wise
            int nonZero = numNeighborsView[y];  
            for (var z = 0; z < nonZero; z++) {
                int colIdx = neighborsView[y, z];
                dot += pView[x, colIdx] * aView[x, colIdx] * edgeWeightsView[y, z];
            }
            outView[index] = dot;
        }


        #endregion

       
    }
}