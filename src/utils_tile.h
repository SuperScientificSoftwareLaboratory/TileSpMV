
#include "common.h"



void swap_key_tile(int *a , int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
/*
void swap_key_tile_arr(int *a , int *b)
{
    for(int i=0;i<16;i++)
    {
        int tmp = *a+i;
        *a+i = *b+i;
        *b+i = tmp;
    }

}*/

// quick sort key (child function)
int partition_key_tile(int *key, int length, int pivot_index , int *nnz, int *csr_ptr)
{
    int i  = 0 ;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key_tile(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap_key_tile(&nnz[pivot_index], &nnz[pivot_index + (length - 1)]);
    for(int k=0;k<16;k++) 
    {
        swap_key_tile(&csr_ptr[pivot_index*16+k], &csr_ptr[(pivot_index + (length - 1))*16+k]);
    }
    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap_key_tile(&key[pivot_index+i], &key[small_length]);
            swap_key_tile(&nnz[pivot_index+i], &nnz[small_length]);
    //        swap_key_tile_arr(&csr_ptr[(pivot_index+i)*16], &csr_ptr[16*small_length]);
            for(int k=0;k<16;k++) 
            {
                swap_key_tile(&csr_ptr[(pivot_index+i)*16+k], &csr_ptr[16*small_length+k]);
            }
            small_length++;
        }
    }

    swap_key_tile(&key[pivot_index + length - 1], &key[small_length]);
    swap_key_tile(&nnz[pivot_index + length - 1], &nnz[small_length]);
  //  swap_key_tile_arr(&csr_ptr[(pivot_index + length - 1)*16], &csr_ptr[16*small_length]);
    for(int k=0;k<16;k++) 
    {
        swap_key_tile(&csr_ptr[(pivot_index + length - 1)*16+k], &csr_ptr[16*small_length+k]);
    }
    return small_length;
}

// quick sort key (main function)
void quick_sort_key_tile(int *key, int length, int *nnz, int *csr_ptr)
{
    if(length == 0 || length == 1)
        return;

    int small_length = partition_key_tile(key, length, 0, nnz, csr_ptr) ;
    quick_sort_key_tile(key, small_length, nnz, csr_ptr);
    quick_sort_key_tile(&key[small_length + 1], length - small_length - 1, &nnz[small_length + 1], &csr_ptr[(small_length + 1)*16]);
    //printf("sort: ")
}

/*
void exclusive_scan_char(unsigned char *input, int length)
{
    if(length == 0 || length == 1)
        return;
    
    unsigned char old_val, new_val;
    
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}*/

