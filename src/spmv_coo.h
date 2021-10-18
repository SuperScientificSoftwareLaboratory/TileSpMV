__global__
void spmv_coo( int*         d_csrRowPtr,
                           int*         d_csrColIdx,
                           MAT_VAL_TYPE*  d_csrVal,
                           int*	d_coo_new_rowidx,
                           int          m, int          new_row_1,
                           MAT_VAL_TYPE*        d_x,
                          MAT_VAL_TYPE*        d_y)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < new_row_1) //new_row_1
    {
//printf("global_id=%d\n",global_id);
       int rowid = d_coo_new_rowidx[global_id];
         int start = d_csrRowPtr[global_id];
        int stop  = d_csrRowPtr[global_id+1];
        MAT_VAL_TYPE sum = 0;
       //  if (stop - start <= LONGROW_THRESHOLD)
      //  {
            for (int j = start; j < stop; j++)
            {
                sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
//printf("global_id=%d  rowid=%d j=%d d_csrColIdx[j]=%d  d_x[d_csrColIdx[j]]=%f  d_csrVal[j]=%f  sum=%f\n",global_id,rowid,j,d_csrColIdx[j],d_x[d_csrColIdx[j]],d_csrVal[j],sum);
            }
      //  }
//printf("global_id=%d  rowid=%d  d_y[rowid]=%f  sum=%f\n",global_id,rowid,d_y[rowid],sum);
         d_y[rowid] += sum;


    }
__syncthreads();
}
