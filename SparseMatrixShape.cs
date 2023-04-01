// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                           Copyright (c) 2023 ILGPU Project
//                                    www.ilgpu.net
//
// File: SparseMatrixShape.cs
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
    /// Represents the shape of a sparse matrix without storing its values.
    /// </summary>
    public class SparseMatrixShape
    {
        #region Nested Types

        /// <summary>
        /// An internal predicate based on mask arrays.
        /// </summary>
        /// <typeparam name="T">The mask element type.</typeparam>
        protected internal readonly struct MaskPredicate<T> :
            InlineList.IPredicate<Index2D>
            where T : struct, IEquatable<T>
        {
            private readonly T[,] mask;
            private readonly T emptyValue;

            /// <summary>
            /// Creates a new mask predicate.
            /// </summary>
            /// <param name="maskArray">The mask array to use.</param>
            /// <param name="emptyValueConstant">
            /// The masking constant to compare each element to.
            /// </param>
            public MaskPredicate(T[,] maskArray, T emptyValueConstant)
            {
                mask = maskArray;
                emptyValue = emptyValueConstant;
            }

            /// <summary>
            /// Returns true if the current mask element is not equal to the empty mask
            /// value.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool Apply(Index2D item) => !mask[item.X, item.Y].Equals(emptyValue);
        }

        /// <summary>
        /// An internal predicate based on float values.
        /// </summary>
        internal readonly struct FloatEpsPredicate :
            InlineList.IPredicate<Index2D>
        {
            private readonly float[,] values;
            private readonly float eps;

            /// <summary>
            /// Creates a new float masking predicate.
            /// </summary>
            /// <param name="valueArray">The input value array.</param>
            /// <param name="epsConstant">
            /// The eps constant to compare each element to.
            /// </param>
            public FloatEpsPredicate(float[,] valueArray, float epsConstant)
            {
                values = valueArray;
                eps = epsConstant;
            }

            /// <summary>
            /// Returns true if the absolute value of the stored mask element is greater
            /// than the epsilon constant.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool Apply(Index2D item) => Math.Abs(values[item.X, item.Y]) > eps;
        }


        #endregion

        #region Static

        /// <summary>
        /// Construct a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="a">A dense MxN matrix.</param>
        /// <param name="eps">The epsilon value used for detecting empty cells.</param>
        /// <returns>A sparse representation of A.</returns>
        public static SparseMatrixShape GetShape<T>(T[,] a, T eps = default)
            where T : struct, IEquatable<T>
        {
            var predicate = new MaskPredicate<T>(a, eps);
            return GetShape(a, predicate);
        }

        /// <summary>
        /// Construct a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="a">A dense MxN matrix.</param>
        /// <param name="eps">The epsilon value used for detecting empty cells.</param>
        /// <returns>A sparse shape representation of A.</returns>
        public static SparseMatrixShape GetShape(float[,] a, float eps = 0.0f)
        {
            var predicate = new FloatEpsPredicate(a, eps);
            return GetShape(a, predicate);
        }

        /// <summary>
        /// Construct a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="a">A dense MxN matrix.</param>
        /// <param name="predicate">The sparse predicate.</param>
        /// <returns>A sparse shape representation of A.</returns>
        public static SparseMatrixShape GetShape<T, TPredicate>(
            T[,] a,
            TPredicate predicate)
            where T : struct, IEquatable<T>
            where TPredicate : InlineList.IPredicate<Index2D>
        {
            var dimensions = GetShape(
                a,
                predicate,
                out int[,] neighbors,
                out int[] numNeighbors,
                out int maxNonZeroEntries);
            return new SparseMatrixShape(
                neighbors,
                numNeighbors,
                dimensions.NumRows,
                dimensions.NumColumns,
                maxNonZeroEntries);
        }

        /// <summary>
        /// Construct a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="a">A dense MxN matrix.</param>
        /// <param name="predicate">The sparse predicate.</param>
        /// <param name="neighbors">The neighbor map.</param>
        /// <param name="numNeighbors">The number of neighbors per row.</param>
        /// <param name="maxNonZeroEntries">
        /// The maximum number of non-zero elements per row.
        /// </param>
        /// <returns>A sparse shape representation of A.</returns>
        private static (int NumRows, int NumColumns) GetShape<T, TPredicate>(
            T[,] a,
            TPredicate predicate,
            out int[,] neighbors,
            out int[] numNeighbors,
            out int maxNonZeroEntries)
            where T : struct, IEquatable<T>
            where TPredicate : InlineList.IPredicate<Index2D>
        {
            int numRows = a.GetLength(0);
            int numColumns = a.GetLength(1);
            var outputNumNeighbors = numNeighbors = new int[numRows];

            // Get counts of the number of columns we need to represent
            int maxNumEntries = 0;
            Parallel.For(0, numRows, i =>
            {
                int numRowEntries = 0;
                for (int j = 0; j < numColumns; ++j)
                {
                    if (predicate.Apply(new Index2D(i, j)))
                        ++numRowEntries;
                }
                outputNumNeighbors[i] = numRowEntries;
                Atomic.Max(ref maxNumEntries, numRowEntries);
            });

            if (maxNumEntries < 1)
                throw new ArgumentOutOfRangeException(nameof(a));
            maxNonZeroEntries = maxNumEntries;

            // Copy non-zero data to sparse storage
            var outputNeighbors = neighbors = new int[numRows, maxNonZeroEntries];

            Parallel.For(0, numRows, i =>
            {
                int index = 0;
                for (int j = 0; j < numColumns; ++j)
                {
                    if (!predicate.Apply(new Index2D(i, j)))
                        continue;

                    outputNeighbors[i, index++] = j;
                }
            });
            return (numRows, numColumns);
        }

        #endregion

        #region Instance

        /// <summary>
        /// NumRows x f matrix containing column indexes where non-zero values in matrix
        /// are for each row in [0:NumRows].
        /// </summary>
        public int[,] neighbors { get; }

        /// <summary>
        /// Vector with number of non-zero entries on each row of m_neighbors for all x,
        /// neighbors[x, numNeighbors[x]:MaxNonZeroEntries] may contain junk.
        /// </summary>
        public int[] numNeighbors { get; }

        /// <summary>
        /// Constructs a new sparse shape using a predefined shape.
        /// </summary>
        /// <param name="other">The source shape.</param>
        protected SparseMatrixShape(SparseMatrixShape other)
            : this(
                other.neighbors,
                other.numNeighbors,
                other.NumRows,
                other.NumColumns,
                other.MaxNonZeroEntries)
        { }

        /// <summary>
        /// Constructs a new sparse shape using explicit information.
        /// </summary>
        private SparseMatrixShape(
            int[,] neighborsArray,
            int[] numNeighborsArray,
            int numRows,
            int numColumns,
            int maxNonZeroEntries)
        {
            neighbors = neighborsArray;
            numNeighbors = numNeighborsArray;
            NumRows = numRows;
            NumColumns = numColumns;
            MaxNonZeroEntries = maxNonZeroEntries;
        }

        #endregion

        #region Properties

        /// <summary>
        /// The number of rows.
        /// </summary>
        public Index1D NumRows { get; }

        /// <summary>
        /// The number of columns.
        /// </summary>
        public Index1D NumColumns { get; }

        /// <summary>
        /// The max # of non-zero entries per row.
        /// </summary>
        public Index1D MaxNonZeroEntries { get; }

        /// <summary>
        /// Emulate GetLength method from array.
        /// </summary>
        /// <param name="dim">Dimension to retrieve information about.</param>
        /// <returns>Length of that dimension</returns>
        public Index1D GetLength(Index1D dim) =>
            dim.X switch
            {
                0 => NumRows,
                1 => NumColumns,
                _ => throw new ArgumentOutOfRangeException(nameof(dim))
            };

        #endregion

        #region Methods

        /// <summary>
        /// Gets the i-th neighbor of the given column.
        /// </summary>
        /// <param name="row">The row to get the neighbor for.</param>
        /// <param name="index">The index of the i-th neighbor to get.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetNeighbor(Index1D row, Index1D index) =>
            neighbors[row, index];

        /// <summary>
        /// Gets the number of neighbors of the given column.
        /// </summary>
        /// <param name="column">The column to get the number of neighbors for.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetNumNeighbors(Index1D column) => numNeighbors[column];

        /// <summary>
        /// Finds the requested column from the original dense matrix in neighbors.
        /// </summary>
        /// <param name="row">The row used for searching.</param>
        /// <param name="column">The column to look for.</param>
        /// <param name="index">The output index (if any).</param>
        /// <returns>True if the column index could be found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected bool TryFindColumn(Index1D row, Index1D column, out Index1D index)
        {
            int nonZero = numNeighbors[row];

            // Simple binary search algorithm
            int left = 0;
            int right = nonZero - 1;
            while (left <= right)
            {
                index = left + (right - left) / 2;
                switch (neighbors[row, index].CompareTo(column))
                {
                    case -1: left = index + 1; break;
                    case 0: return true;
                    case 1: right = index - 1; break;
                }
            }
            index = Index1D.Invalid;
            return false;
        }

        #endregion
    }
}
