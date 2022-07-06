#include"common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


# define INDEX_DATA_TYPE unsigned char
# define VAL_DATA_TYPE double

void encode_new(INDEX_DATA_TYPE *str1, INDEX_DATA_TYPE *str2, INDEX_DATA_TYPE *res, int str_len)
{
    int count = 0;
    for (int i = 0; i < str_len; i++)
    {
        res[count] = (str1[i] << 4) + str2[i];
        count++;
    }
}

void decode_new(INDEX_DATA_TYPE *str1, INDEX_DATA_TYPE *str2, INDEX_DATA_TYPE *res, int res_len)
{
    for (int i = 0; i < res_len; i++)
    {
        str1[i] = (res[i] & num_f) >> 4;
        str2[i] =  res[i] & num_b;
    }
}

void encode(INDEX_DATA_TYPE *str, INDEX_DATA_TYPE *res, int str_len,int offset)
{
    if (str_len % 2 == 0)
    {
        int count = offset;
        for (int i = 0; i < str_len; i = i + 2)
        {
            res[count] = (str[i] << 4) + str[i + 1];
            count++;
        }
    }
    else
    {
        int count = offset;
        for (int i = 0; i < str_len - 1; i = i + 2)
        {
            res[count] = (str[i] << 4) + str[i + 1];
            count++;
        } 
        res[count] = str[str_len - 1] << 4;
    }
}

void decode(INDEX_DATA_TYPE *str, INDEX_DATA_TYPE *res, int res_len)
{

    if (res_len % 2 == 0)
    {
        for (int i = 0; i < res_len; i = i + 2)
        {
            res[i] = (str[i / 2] & num_f) >> 4;
            res[i + 1] =  str[i / 2] & num_b;
        }
    }
    else
    {
        for (int i = 0; i < res_len - 1; i = i + 2)
        {
            res[i] = (str[i / 2] & num_f) >> 4;
            res[i + 1] =  str[i / 2] & num_b;
        }
        res[res_len - 1] = str[res_len / 2] >> 4;
    }
}

void transposition_CSR_to_COO(INDEX_DATA_TYPE *csr_rowPtr, INDEX_DATA_TYPE *csr_colIdx, VAL_DATA_TYPE *csr_val,
                              INDEX_DATA_TYPE *coo_rowIdx, INDEX_DATA_TYPE *coo_colIdx, VAL_DATA_TYPE *coo_val,
                              int m, int nnz)
{
    int count = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = csr_rowPtr[i]; j < csr_rowPtr[i + 1]; j++)
        {
            coo_rowIdx[count] = i;
            count++;
        }
    }
    memcpy(coo_colIdx, csr_colIdx, sizeof(INDEX_DATA_TYPE) * nnz);
    memcpy(coo_val, csr_val, sizeof(VAL_DATA_TYPE) * nnz);
    return;
}

void transposition_CSR_to_ELC(INDEX_DATA_TYPE *csr_rowPtr, INDEX_DATA_TYPE *csr_colIdx, VAL_DATA_TYPE *csr_val,
                              INDEX_DATA_TYPE *elc_colIdx, VAL_DATA_TYPE *elc_val, int m, int max_width)
{
    memset(elc_colIdx, 0, sizeof(INDEX_DATA_TYPE) * m * max_width);
    memset(elc_val, 0, sizeof(VAL_DATA_TYPE) * m * max_width);
    for (int i = 0; i < m; i++)
    {
        for (int j = csr_rowPtr[i]; j < csr_rowPtr[i + 1]; j++)
        {
            int temp = j - csr_rowPtr[i];
            elc_colIdx[temp * m + i] = csr_colIdx[j];
            elc_val[temp * m + i] = csr_val[j];
        }
    }
}

void transposition_CSR_to_DENSE(INDEX_DATA_TYPE *csr_rowPtr, INDEX_DATA_TYPE *csr_colIdx, VAL_DATA_TYPE *csr_val,
                                VAL_DATA_TYPE *dense_val, int m, int n)
{
    memset(dense_val, 0, sizeof(VAL_DATA_TYPE) * m * n);
    for (int i = 0; i < m; i++)
    {
        for (int j = csr_rowPtr[i]; j < csr_rowPtr[i+1]; j++)
        {
            dense_val[i * n + csr_colIdx[j]] = csr_val[j];
        }
    }
    return;
}

