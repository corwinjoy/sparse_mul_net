// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                           Copyright (c) 2023 ILGPU Project
//                                    www.ilgpu.net
//
// File: SparseMatrix.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using ILGPU;
using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace MatrixMultiplySparse
{
    /// <summary>
    /// Represents a sparse matrix in CPU space.
    /// </summary>
    public class SparseMatrix : SparseMatrixShape
    {
        #region Static

        /// <summary>
        /// Construct a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="a">A dense MxN matrix.</param>
        /// <param name="eps">The epsilon value used for detecting empty cells.</param>
        /// <returns>A sparse shape representation of A.</returns>
        public static SparseMatrix Create(float[,] a, float eps = 0.0f)
        {
            var predicate = new FloatEpsPredicate(a, eps);
            return Create(a, predicate);
        }

        /// <summary>
        /// Construct a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="a">A dense MxN matrix.</param>
        /// <param name="predicate">The sparse predicate.</param>
        /// <returns>A sparse shape representation of A.</returns>
        public static SparseMatrix Create<TPredicate>(
            float[,] a,
            TPredicate predicate)
            where TPredicate : InlineList.IPredicate<Index2D>
        {
            var shape = GetShape(a, predicate);

            var edgeWeights = new float[shape.NumRows, shape.MaxNonZeroEntries];
            Parallel.For(0, shape.NumRows, i =>
            {
                int index = 0;
                for (int j = 0; j < shape.NumColumns; ++j)
                {
                    if (!predicate.Apply(new Index2D(i, j)))
                        continue;

                    edgeWeights[i, index++] = a[i, j];
                }
            });
            return new SparseMatrix(shape, edgeWeights);
        }

        #endregion

        #region Instance

        /// <summary>
        /// Weights for each entry in neighbors.
        /// </summary>
        private readonly float[,] edgeWeights;

        /// <summary>
        /// Constructs a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="shape">A sparse shape.</param>
        /// <param name="edgeWeightsArray">The values for all edges.</param>
        protected SparseMatrix(SparseMatrixShape shape, float[,] edgeWeightsArray)
            : base(shape)
        {
            edgeWeights = edgeWeightsArray;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets a data element in the specified row and column.
        /// </summary>
        public float this[Index1D row, Index1D column]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (!TryFindColumn(row, column, out var idx))
                    return 0.0f;
                return DirectAccess(row, idx);
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                if (!TryFindColumn(row, column, out var idx))
                    throw new ArgumentOutOfRangeException(nameof(column));
                DirectAccess(row, idx) = value;
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Returns the underlying edge weights array.
        /// </summary>
        internal float[,] GetEdgeWeights() => edgeWeights;

        /// <summary>
        /// Returns a memory reference to the desired data cell.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref float DirectAccess(Index1D row, Index1D idx) =>
            ref edgeWeights[row, idx];

        #endregion
    }
}
