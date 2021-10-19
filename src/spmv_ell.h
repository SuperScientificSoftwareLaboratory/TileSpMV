__inline__ __device__
int spmv_ell(int rbnum, int cbnum, int rowA, int colA, int blkj,int blki,int lane_id,int collength,
char *d_blkwidth, MAT_VAL_TYPE *d_Blockell_Val, unsigned char *d_ell_colIdx, MAT_VAL_TYPE *d_x,
int * d_ell_offset, int x_offset ,MAT_VAL_TYPE sum) 
{

                sum = 0;
int ellwoffset = d_ell_offset[blkj];
                // load x needed by this block to REGISTER
                MAT_VAL_TYPE r_x = lane_id < collength ? d_x[x_offset + lane_id] : 0;

                // produce all intermediate products
                int elllen = d_blkwidth[blkj] * BLOCK_SIZE;

                for (int rj = lane_id; rj < elllen; rj += WARP_SIZE)
                {
int ellcolidx = d_ell_colIdx[ellwoffset + rj];
MAT_VAL_TYPE r_x_gathered =d_x[x_offset + ellcolidx];//s_x_warp[ellcolidx];
                    sum += d_Blockell_Val[ellwoffset + rj] * r_x_gathered;
                }
                // fuse sum at virtual_lane_id and it+16
                sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

return sum;

            }
