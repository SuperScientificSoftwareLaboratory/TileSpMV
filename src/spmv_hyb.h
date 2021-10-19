__inline__ __device__
int spmv_hyb(int rbnum, int cbnum, int rowA, int colA, int blkj,int blki,int lane_id,int collength,
char *d_blkwidth, unsigned char *d_hybIdx, MAT_VAL_TYPE *d_Blockhyb_Val,  MAT_VAL_TYPE *d_x,
int * d_hyb_offset, int x_offset ,MAT_VAL_TYPE sum) 
{

                sum = 0;
int hybwoffset = d_hyb_offset[blkj];

                MAT_VAL_TYPE r_x = lane_id < collength ? d_x[x_offset + lane_id] : 0;

                int hyblen = d_blkwidth[blkj] * BLOCK_SIZE;

                for (int rj = lane_id; rj < hyblen; rj += WARP_SIZE)
                {
int hybcolidx = d_hybIdx[hybwoffset + rj];


MAT_VAL_TYPE r_x_gathered =d_x[x_offset + hybcolidx];

                    sum += d_Blockhyb_Val[hybwoffset + rj] * r_x_gathered;

                }
                sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

return sum;

            }
