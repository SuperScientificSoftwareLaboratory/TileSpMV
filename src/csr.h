__inline__ __device__
int spmv_csr(int rbnum, int cbnum, int rowA, int colA, int blkj,int blki,int lane_id,
unsigned char *d_csr_compressedIdx, MAT_VAL_TYPE *d_Blockcsr_Val, unsigned char *d_Blockcsr_Ptr,MAT_VAL_TYPE *d_x,
int * d_csr_offset, int * d_csrptr_offset ,int x_offset ,int sum) 
{

int csroffset = d_csr_offset[blkj ];
 int csrcount = d_csrptr_offset[blkj];

                  int ri  = lane_id >> 1; // / 2;
                int virtual_lane_id = lane_id & 0x1; // % 2;

int rowlength= blki==rbnum-1 ? rowA-(rbnum-1)*BLOCK_SIZE : BLOCK_SIZE ;
int stop=0;
if(lane_id<rowlength||lane_id-16<rowlength)
 stop = ri == BLOCK_SIZE - 1 ? d_csr_offset[blkj+1] -csroffset: d_Blockcsr_Ptr[ri+1+csrcount];//+1 buyidingdui
if(rowA-1==ri+blki*BLOCK_SIZE)
stop=d_csr_offset[blkj+1] -csroffset;

int start=d_Blockcsr_Ptr[csrcount + ri] + virtual_lane_id;

              for (int rj = start; rj < stop; rj += 2)
             {
                 if(lane_id<rowlength||lane_id-16<rowlength)
                     sum += d_x[x_offset + d_csr_compressedIdx[csroffset + rj]] * d_Blockcsr_Val[csroffset + rj];
             }
                // fuse sum at virtual_lane_id 0 and 1
                 sum += __shfl_down_sync(0xffffffff, sum, 1);

return sum;

            }
