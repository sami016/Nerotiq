/**
 * Matrix structure.
 */
struct Matrix {
	/**
	 * The number of rows.
	 */
	ushort Height;
	/**
	 * The number of columns.
	 */
    ushort Width;
	/**
	 * Data array pointer.
	 */
    float *Data;
};

/**
 * Gets the index within a matrix array for a given row and column.
 */
inline int getMatrixIndex(struct Matrix matrix, uint row, uint column) {
    return row * matrix.Width + column;
}

/**
 * Gets the entry of a matrix at a given row and column.
 */
inline float getMatrixValue(struct Matrix matrix, uint row, uint column) {
    return matrix.Data[getMatrixIndex(matrix, row, column)];
}

/**
 * Sets the entry of a matrix at a given row and column.
 */
inline void setMatrixValue(struct Matrix matrix, uint row, uint column, double value) {
	matrix.Data[getMatrixIndex(matrix, row, column)] = value;
}
