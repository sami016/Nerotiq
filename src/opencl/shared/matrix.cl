
/**
 * Gets the entry of a matrix.
 */
inline float getMatrixValue(uint row, uint column, uint rowWidth) {
    return row * rowWidth + column;
}