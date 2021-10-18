
#include"spmv_csr.h"
#include"spmv_coo.h"
#include"spmv_ell.h"
#include"spmv_hyb.h"
#include"spmv_dense.h"
#include"spmv_dense_row.h"
#include"spmv_dense_col.h"

__global__ 
void stir_spmv_cuda_kernel_v6(int rbnum, int cbnum, int rowA, int colA,int dense_size,
                        int *d_nnzb_A, char *d_flag, MAT_PTR_TYPE *d_rowblock_ptr, int *d_columnid, char * d_Format,int *d_blknnz, unsigned char *d_blknnznnz,

                        unsigned char *d_csr_compressedIdx, MAT_VAL_TYPE *d_Blockcsr_Val, unsigned char *d_Blockcsr_Ptr, 

                        char *d_blkwidth, MAT_VAL_TYPE *d_Blockell_Val, unsigned char *d_ell_colIdx, 

                        unsigned char *d_hybIdx, MAT_VAL_TYPE *d_Blockhyb_Val, 
 
                        MAT_VAL_TYPE *d_Blockdense_Val, 

                        int * d_denserowptr, MAT_VAL_TYPE *d_Blockdenserow_Val, char *d_denserowid, 

                        int * d_densecolptr, MAT_VAL_TYPE *d_Blockdensecol_Val, char *d_densecolid,  

                        int * d_dns_offset, int * d_dnsrow_offset, int * d_dnscol_offset, int * d_csr_offset, int * d_csrptr_offset, int * d_ell_offset, int * d_coo_offset, int * d_hyb_coocount, int * d_hyb_offset,

                        int rowblkblock, unsigned int * d_blkcoostylerowidx, int * d_blkcoostylerowidx_colstart, int * d_blkcoostylerowidx_colstop,

                        int *d_multicoo_ptr, int *d_multicoo_colidx, MAT_VAL_TYPE *d_multicoo_val,int *d_coo_new_matrix_ptr,

                        MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y, int formatprofile, int *d_coodeferoffset, int *d_deferbuf_coooff, int *d_deferbuf_dxoff)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
   // printf("global_id=%d\n",global_id);
    const int blki_blc = global_id >> 5; // WARP_SIZE;


    __shared__ MAT_VAL_TYPE s_x[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ MAT_VAL_TYPE s_y[WARP_PER_BLOCK * BLOCK_SIZE];

    const int local_warp_id = threadIdx.x >> 5; // / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    MAT_VAL_TYPE *s_x_warp = &s_x[local_warp_id * BLOCK_SIZE];
    MAT_VAL_TYPE *s_y_warp = &s_y[local_warp_id * BLOCK_SIZE];

    __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ char s_Format[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    char *s_Format_local = &s_Format[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ unsigned char s_blknnznnz[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    unsigned char *s_blknnznnz_local = &s_blknnznnz[local_warp_id * PREFETCH_SMEM_TH];

    __shared__ int s_dns_offset[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_dns_offset_local = &s_dns_offset[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_dnsrow_offset[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_dnsrow_offset_local = &s_dnsrow_offset[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_dnscol_offset[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_dnscol_offset_local = &s_dnscol_offset[local_warp_id * PREFETCH_SMEM_TH];

    __shared__ int s_csr_offset[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_csr_offset_local = &s_csr_offset[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_ell_offset[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ell_offset_local = &s_ell_offset[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_coo_offset[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_coo_offset_local = &s_dns_offset[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_hyb_offset[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_hyb_offset_local = &s_hyb_offset[local_warp_id * PREFETCH_SMEM_TH];

    __shared__ int s_csrptr_offset[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_csrptr_offset_local = &s_csrptr_offset[local_warp_id * PREFETCH_SMEM_TH];


    if (blki_blc < rowblkblock)
    {
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;

        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;

        MAT_VAL_TYPE sum = 0;
        MAT_VAL_TYPE sumsum = 0;
        if (lane_id < BLOCK_SIZE)
            s_y_warp[lane_id] = 0;

        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_rowblock_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_rowblock_ptr[blki+1];

        if (lane_id < rowblkjstop - rowblkjstart)
        {
            s_columnid_local[lane_id] = d_columnid[rowblkjstart + lane_id];
            s_Format_local[lane_id] = d_Format[rowblkjstart + lane_id];
            s_blknnznnz_local[lane_id] = d_blknnznnz[rowblkjstart + lane_id];

            s_dns_offset_local[lane_id] = d_dns_offset[rowblkjstart + lane_id];
            s_dnsrow_offset_local[lane_id] = d_dnsrow_offset[rowblkjstart + lane_id];
            s_dnscol_offset_local[lane_id] = d_dnscol_offset[rowblkjstart + lane_id];
            s_csr_offset_local[lane_id] = d_csr_offset[rowblkjstart + lane_id];
            s_ell_offset_local[lane_id] = d_ell_offset[rowblkjstart + lane_id];
            s_coo_offset_local[lane_id] = d_coo_offset[rowblkjstart + lane_id];
            s_hyb_offset_local[lane_id] = d_hyb_offset[rowblkjstart + lane_id];
 
            s_csrptr_offset_local[lane_id] = d_csrptr_offset[rowblkjstart + lane_id];
   
        }
        int coodeferoffset = 0;

        // for each block in the block row

        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj++)
        {

	    char subformat = s_Format_local[blkj - rowblkjstart];
            int colid = s_columnid_local[blkj - rowblkjstart];
 
            int collength = colid == cbnum-1 ? colA - (cbnum-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            int x_offset = colid * BLOCK_SIZE;

#if DEBUG_FORMATCOST
        if (formatprofile != 7) //
#endif
           switch(subformat) 
            {
              case 0:
#if DEBUG_FORMATCOST
            if (formatprofile == 0 || formatprofile == -1)
#endif
            {
                sum = 0;
sum=spmv_csr(rbnum, cbnum, rowA, colA, blkj, blki, lane_id,
d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,d_x,
d_csr_offset, d_csrptr_offset,x_offset,sum  ) ;

                // move sum usable to the first 16 lanes
                sumsum += __shfl_down_sync(0xffffffff, sum, lane_id);
            }
            
            break;
            case 1:
            // if COO
            // the reason of using COO is that the number of nonzeros is very small (no more than 32?), 
            // and the nonzeros should be distributed evenly among the rows (otherwise ELL should be used), 
            // so all nonzeros could be loaded to registers, and atomic add the y to smem
//            else if ((subformat == 1 && formatprofile == 1) || (subformat == 1 && formatprofile == -1))
#if DEBUG_FORMATCOST
            if (formatprofile == 1 || formatprofile == -1)
#endif

            break;
            case 2:
            // if ELL 
            // (could use register shuffle for gathering x (when width is multiple of 2), 
            //  since all threads should be active when shuffling (otherwise 
            //  __shfl_sync gives undefined output))
//            else if ((subformat == 2 && formatprofile == 2) || (subformat == 2 && formatprofile == -1))
#if DEBUG_FORMATCOST
            if (formatprofile == 2 || formatprofile == -1)
#endif
            {
                sum = 0;
sum=spmv_ell(rbnum, cbnum, rowA, colA, blkj, blki, lane_id,collength,
d_blkwidth, d_Blockell_Val, d_ell_colIdx,d_x,
d_ell_offset,x_offset,sum  ) ;

                sumsum += sum;
            }
            
            break;

            case 3:
            // if HYB
//            else if ((subformat == 3 && formatprofile == 3) || (subformat == 3 && formatprofile == -1))
#if DEBUG_FORMATCOST
            if (formatprofile == 3 || formatprofile == -1)
#endif
{
                sum = 0;

sum=spmv_hyb(rbnum, cbnum, rowA, colA, blkj,blki,lane_id, collength,
d_blkwidth, d_hybIdx, d_Blockhyb_Val,  d_x,
 d_hyb_offset,  x_offset , sum);

                sumsum += sum;
}
            break;
      

            case 4:
            // if dense (or near dense stored as dense)
            // load x into registers, and do gemm as usual 
            // (the dense block should be stored in column-major)
//            else if ((subformat == 4 && formatprofile == 4) || (subformat == 4 && formatprofile == -1))
#if DEBUG_FORMATCOST
            if (formatprofile == 4 || formatprofile == -1)
#endif
            {
                sum = 0;

sum=spmv_dense(rbnum, cbnum, rowA, colA, blkj, blki, lane_id, dense_size,
d_Blockdense_Val, d_x,
d_dns_offset, x_offset, sum);

                sumsum += sum;

            }
            
            break;
             case 5:
           // if dense row (only work when the #nonzeros in the block is multiple of BLOCK_SIZE)
            // do dot product, and add the result into sum
//            else if ((subformat == 5 && formatprofile == 5) || (subformat == 5 && formatprofile == -1))
#if DEBUG_FORMATCOST
            if (formatprofile == 5 || formatprofile == -1)
#endif
            {

spmv_dense_row(rbnum, cbnum, rowA, colA, blkj, blki, lane_id, dense_size, rowblkjstart, collength,
d_denserowptr, d_Blockdenserow_Val, d_denserowid,  d_x, s_y_warp,
s_dnsrow_offset_local, x_offset, sum);
			}
			
            break;
            case 6:
            // if dense col (only work when the #nonzeros in the block is multiple of BLOCK_SIZE)
            // do scaling, and add the result into sum
//            else if ((subformat == 6 && formatprofile == 6) || (subformat == 6 && formatprofile == -1))
#if DEBUG_FORMATCOST
            if (formatprofile == 6 || formatprofile == -1)
#endif
            {
                sum = 0;

sum=spmv_dense_col(rbnum, cbnum, rowA, colA, blkj, blki, lane_id, dense_size, rowblkjstart, collength,
d_densecolptr, d_Blockdensecol_Val, d_densecolid, d_x, s_y_warp,
s_dnscol_offset_local, x_offset, sum);

			    sumsum += sum;
            }
            
            break;
            }
        }

        if (lane_id < BLOCK_SIZE)
            sumsum += s_y_warp[lane_id];

        // save sum to d_y
        if (lane_id < BLOCK_SIZE && sumsum != 0)
        {
            if (signbit)
                atomicAdd(&d_y[blki * BLOCK_SIZE + lane_id], sumsum);
            else
                d_y[blki * BLOCK_SIZE + lane_id] = sumsum;
        }
    }


}


