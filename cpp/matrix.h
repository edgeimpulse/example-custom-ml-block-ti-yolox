#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stddef.h>
#ifdef __cplusplus
#include <functional>
#endif // __cplusplus

typedef struct ei_matrix {
    float *buffer;
    uint32_t rows;
    uint32_t cols;
    bool buffer_managed_by_me;

#ifdef __cplusplus
    /**
     * Create a new matrix
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @param a_buffer Buffer, if not provided we'll alloc on the heap
     */
    ei_matrix(
        uint32_t n_rows,
        uint32_t n_cols,
        float *a_buffer = NULL
        )
    {
        if (a_buffer) {
            buffer = a_buffer;
            buffer_managed_by_me = false;
        }
        else {
            buffer = (float*)calloc(n_rows * n_cols * sizeof(float), 1);
            buffer_managed_by_me = true;
        }
        rows = n_rows;
        cols = n_cols;
    }

    ~ei_matrix() {
        if (buffer && buffer_managed_by_me) {
            free(buffer);
        }
    }

    /**
     * @brief Get a pointer to the buffer advanced by n rows
     *
     * @param row Numer of rows to advance the returned buffer pointer
     * @return float* Pointer to the buffer at the start of row n
     */
    float *get_row_ptr(size_t row)
    {
        return buffer + row * cols;
    }

#endif // #ifdef __cplusplus
} matrix_t;

typedef struct ei_matrix_i32 {
    int32_t *buffer;
    uint32_t rows;
    uint32_t cols;
    bool buffer_managed_by_me;

#ifdef __cplusplus
    /**
     * Create a new matrix
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @param a_buffer Buffer, if not provided we'll alloc on the heap
     */
    ei_matrix_i32(
        uint32_t n_rows,
        uint32_t n_cols,
        int32_t *a_buffer = NULL
        )
    {
        if (a_buffer) {
            buffer = a_buffer;
            buffer_managed_by_me = false;
        }
        else {
            buffer = (int32_t*)calloc(n_rows * n_cols * sizeof(int32_t), 1);
            buffer_managed_by_me = true;
        }
        rows = n_rows;
        cols = n_cols;
    }

    ~ei_matrix_i32() {
        if (buffer && buffer_managed_by_me) {
            free(buffer);
        }
    }

    /**
     * @brief Get a pointer to the buffer advanced by n rows
     *
     * @param row Numer of rows to advance the returned buffer pointer
     * @return float* Pointer to the buffer at the start of row n
     */
    int32_t *get_row_ptr(size_t row)
    {
        return buffer + row * cols;
    }

#endif // #ifdef __cplusplus
} matrix_i32_t;

#endif // _MATRIX_H_
