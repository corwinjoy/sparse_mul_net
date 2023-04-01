using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;




namespace MatrixMultiplySparse
{
    public static class Utils
    {
        /// <summary>
        /// Print a matrix to the console
        /// </summary>
        /// <param name="a">A MxN matrix</param>
        public static void PrintMatrix(dynamic a)
        {
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    Console.Write(a[i,j] + "\t");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
        
        
        /// <summary>
        /// Creates a matrix populated with random values.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix</param>
        /// <param name="columns">The number of columns in the matrix</param>
        /// <returns>A matrix populated with random values</returns>
        [SuppressMessage(
            "Security",
            "CA5394:Do not use insecure randomness",
            Justification = "Only used for testing")]
        public static float[,] CreateRandomMatrix(int rows, int columns)
        {
            var rnd = new Random();
            var matrix = new float[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                    matrix[i, j] = rnd.Next(minValue: -100, maxValue: 100);
            }

            return matrix;
        }


        /// <summary>
        /// Creates a matrix populated with sequential values.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix</param>
        /// <param name="columns">The number of columns in the matrix</param>
        /// <returns>A matrix populated with sequential values</returns>
        public static float[,] CreateSequentialMatrix(int rows, int columns)
        {
            float seq = 0.0f;
            var matrix = new float[rows, columns];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++)
                    matrix[i, j] = seq++;
            }

            return matrix;
        }


        /// <summary>
        /// Creates a n-diagonal matrix populated with sequential values.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix</param>
        /// <param name="columns">The number of columns in the matrix</param>
        /// <param name="band_width">Band-width around the diagonal to populate</param>
        /// <returns>A banded diagonal matrix populated with sequential values</returns>
        public static float[,] CreateBandedSequentialMatrix(int rows, int columns, int band_width)
        {
            float seq = 0.0f;
            var matrix = new float[rows, columns];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++) 
                {
                    if(Math.Abs(i-j) < band_width) 
                    {
                        matrix[i, j] = seq++;
                    } else 
                    {
                        matrix[i, j] = 0.0f;
                    }
                }
                    
            }

            return matrix;
        }

        /// <summary>
        /// Creates an Identity matrix.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix</param>
        /// <param name="columns">The number of columns in the matrix</param>
        /// <returns>A matrix populated 1s along the diagonal and 0s elsewhere</returns>
        public static float[,] CreateIndentityMatrix(int rows, int columns)
        {
            var matrix = new float[rows, columns];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++) 
                {
                    if(i==j) 
                    {
                        matrix[i, j] = 1.0f;
                    }
                    else 
                    {
                        matrix[i, j] = 0.0f;
                    }
                }
                    
            }
            return matrix;
        }

        /// <summary>
        /// Compares two matrices for equality.
        /// </summary>
        /// <param name="a">A MxN matrix (the actual matrix we got)</param>
        /// <param name="e">A MxN matrix (the matrix we expected) </param>
        /// <returns>True if the matrices are equal</returns>
        public static bool MatrixEqual(dynamic a, dynamic e)
        {
            var ma = a.GetLength(0);
            var na = a.GetLength(1);
            var me = e.GetLength(0);
            var ne = e.GetLength(1);

            if (ma != me || na != ne)
            {
                Debug.WriteLine($"Matrix dimensions do not match: [{ma}x{na}] vs [{me}x{ne}]");
                return false;
            }

            for (var i = 0; i < ma; i++)
            {
                for (var j = 0; j < na; j++)
                {
                    var actual = a[i, j].ToString("G4");  // G4 = 4 significant digits
                    var expected = e[i, j].ToString("G4");
                    if (actual != expected) 
                    {
                        Debug.WriteLine($"Error at element location [{i}, {j}]: {actual} found, {expected} expected");
                        return false;
                    }
                }
            }

            return true;
        }


        /// <summary>
        /// Multiplies two dense matrices and returns the resultant matrix.
        /// </summary>
        /// <param name="accelerator">The Accelerator to run the multiplication on</param>
        /// <param name="a">A dense MxK matrix</param>
        /// <param name="b">A dense KxN matrix</param>
        /// <returns>A dense MxN matrix</returns>
        public static float[,] MatrixMultiplyNaive(dynamic a, dynamic b)
        {
            var m = a.GetLength(0);
            var ka = a.GetLength(1);
            var kb = b.GetLength(0);
            var n = b.GetLength(1);

            if (ka != kb)
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(b));

            var c = new float[m, n];

            for (var x = 0; x < m; x++)
            {
                for (var y = 0; y < n; y++)
                {
                    c[x, y] = 0;

                    for (var z = 0; z < ka; z++)
                        c[x, y] += a[x, z] * b[z, y];
                }
            }

            return c;
        }

        /// <summary>
        /// Compute the transpose of a matrix.
        /// </summary>
        /// <param name="a">A MxN matrix</param>
        /// <returns>The transpose of A, a NxM matrix</returns>
        public static float[,] MatrixTranspose(float[,] a)
        {
            int rows = (int) a.GetLength(0);
            int columns = (int) a.GetLength(1);
            var a_t = new float[columns, rows];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++)
                {
                    a_t[j, i] = a[i, j];
                }
            }

            return a_t;
        }

        /// <summary>
        /// Apply a binary mask to a matrix
        /// </summary>
        /// <param name="a">A MxN matrix</param>
        /// <param name="mask">A MxN binary mask of 1s and 0s</param>
        /// <returns>(a && mask)  that is, the matrix a as masked by mask</returns>
        public static float[,] MatrixMask(float[,] a, float[,] mask)
        {
            var ma = a.GetLength(0);
            var na = a.GetLength(1);
            var mm = mask.GetLength(0);
            var nm = mask.GetLength(1);

            if (ma != mm || na != nm)
            {
                var err = $"Matrix dimensions do not match: [{ma}x{na}] vs [{mm}x{nm}]";
                Debug.WriteLine(err);
                throw new ArgumentException(err);
            }

            var a_m = new float[ma, na];

            for (var i = 0; i < ma; i++)
            {
                for (var j = 0; j < na; j++)
                {
                    if (mask[i, j] != 0.0f)
                    {
                        a_m[i, j] = a[i, j];
                    } 
                    else 
                    {
                        a_m[i, j] = 0.0f;
                    }  
                }
            }

            return a_m;
        }

        public static int[] To1DArray(int[,] input)
        {
            // Step 1: get total size of 2D array, and allocate 1D array.
            int size = (int) input.Length;
            int[] result = new int[size];
            
            // Step 2: copy 2D array elements into a 1D array.
            int write = 0;
            for (int i = 0; i <= input.GetUpperBound(0); i++)
            {
                for (int z = 0; z <= input.GetUpperBound(1); z++)
                {
                    result[write++] = input[i, z];
                }
            }
            // Step 3: return the new array.
            return result;
        }
    } // end class Utils   

    static class ArraySliceExt
    {
        public static ArraySlice2D<T> Slice<T>(this T[,] arr, int firstDimension, int length)
        {
            return new ArraySlice2D<T>(arr, firstDimension, length);
        }
    }
    class ArraySlice2D<T>
    {
        private readonly T[,] arr;
        private readonly int firstDimension;
        private readonly int length;
        public int Length { get { return length; } }
        public ArraySlice2D(T[,] arr, int firstDimension, int length)
        {
            this.arr = arr;
            this.firstDimension = firstDimension;
            this.length = length;
        }
        public T this[int index]
        {
            get { return arr[firstDimension, index]; }
            set { arr[firstDimension, index] = value; }
        }

        public int BinarySearch(T value)
        {
            int left = 0;
            int right = this.length - 1;
        
            int middle;
            while (left <= right)
            {
                // middle = (left + right) / 2;  Overflow bug
                middle = left + (right-left)/2;
                switch (Compare(this[middle], value))
                {
                    case -1: left = middle + 1; break;
                    case 0: return middle;
                    case 1: right = middle - 1; break;
                }
            }
            throw new ArgumentOutOfRangeException("Element not found");
        }

        private int Compare(T x, T y)
        {
            return Comparer<T>.Default.Compare(x, y);
        }
    }
} // end namespace MatrixMultiply