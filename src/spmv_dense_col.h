__inline__ __device__
int spmv_dense_col(int rbnum, int cbnum, int rowA, int colA, int blkj,int blki,int lane_id,int dense_size,int rowblkjstart,int collength,
 int * d_densecolptr, MAT_VAL_TYPE *d_Blockdensecol_Val, char *d_densecolid,   MAT_VAL_TYPE *d_x,MAT_VAL_TYPE *s_y_warp,
int * s_dnscol_offset_local, int x_offset ,MAT_VAL_TYPE sum) 
{
  int dnscoloffset = s_dnscol_offset_local[blkj - rowblkjstart]; 
                // load x needed by this block to REGISTER
                
                // produce all intermediate products
                int colptrstart = d_densecolptr[blkj];
                int dnswidth = d_densecolptr[blkj + 1] - colptrstart;

int rowlength= blki==rbnum-1 ? rowA-(rbnum-1)*BLOCK_SIZE : BLOCK_SIZE ;
int width=  blki==rbnum-1 ? rowlength*2 : WARP_SIZE ;

                for (int glid = lane_id; glid < dnswidth * BLOCK_SIZE; glid += WARP_SIZE)
                {

                    int rj = glid >> 4; //glid / BLOCK_SIZE;
                    int ri = glid % BLOCK_SIZE;
if(ri>=rowlength) continue;
                    ri += dnscoloffset;
                    ri += rj * rowlength;//BLOCK_SIZE
                    rj += colptrstart; 
                    rj = d_densecolid[rj];
                    rj += x_offset;
if(rj<rowA)  
                    sum += d_Blockdensecol_Val[ri] * d_x[rj];  
                }
                // fuse sum at virtual_lane_id and it+16
                sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

return sum;
            }
