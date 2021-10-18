__inline__ __device__
int spmv_dense(int rbnum, int cbnum, int rowA, int colA, int blkj,int blki,int lane_id,int dense_size,
MAT_VAL_TYPE *d_Blockdense_Val, MAT_VAL_TYPE *d_x,
int * d_dns_offset, int x_offset ,MAT_VAL_TYPE sum) 
{
int denseoffset = d_dns_offset[blkj];
int rowlength= blki==rbnum-1 ? rowA-(rbnum-1)*BLOCK_SIZE : BLOCK_SIZE ;
int width=  blki==rbnum-1 ? rowlength*2 : WARP_SIZE ;
MAT_VAL_TYPE r_x=0;
if(x_offset + lane_id< rowA)
                 r_x = lane_id <  BLOCK_SIZE? d_x[x_offset + lane_id] : 0;//

                // copy the x to both half of the 32 registers
                r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);
                int xoff1 = lane_id >> 4; /// BLOCK_SIZE;

                MAT_VAL_TYPE r_x_gathered;
                int val_offset;
                // round 0
                r_x_gathered = __shfl_sync(0xffffffff, r_x,  xoff1 * BLOCK_SIZE + xoff1);
if(lane_id<rowlength)
{
                val_offset = denseoffset+lane_id;
sum += r_x_gathered * d_Blockdense_Val[val_offset];
} 
else if(lane_id-16<rowlength)
{
    val_offset = denseoffset+lane_id-16+rowlength; 
sum += r_x_gathered * d_Blockdense_Val[val_offset];
}

                r_x_gathered = __shfl_sync(0xffffffff, r_x,  xoff1 * BLOCK_SIZE + xoff1 + 2);
                val_offset += width;
if(val_offset<dense_size)
                sum += r_x_gathered * d_Blockdense_Val[val_offset];
else sum += 0;

                r_x_gathered = __shfl_sync(0xffffffff, r_x,  xoff1 * BLOCK_SIZE + xoff1 + 4);
                val_offset += width;
if(val_offset<dense_size)
                sum += r_x_gathered * d_Blockdense_Val[val_offset];
else sum += 0;

                r_x_gathered = __shfl_sync(0xffffffff, r_x,  xoff1 * BLOCK_SIZE + xoff1 + 6);
                val_offset += width;
if(val_offset<dense_size)
                sum += r_x_gathered * d_Blockdense_Val[val_offset];
else sum += 0;

                r_x_gathered = __shfl_sync(0xffffffff, r_x,  xoff1 * BLOCK_SIZE + xoff1 + 8);
                val_offset += width;
if(val_offset<dense_size)
                sum += r_x_gathered * d_Blockdense_Val[val_offset];
else sum += 0;

                r_x_gathered = __shfl_sync(0xffffffff, r_x,  xoff1 * BLOCK_SIZE + xoff1 + 10);
                val_offset += width;
if(val_offset<dense_size)
                sum += r_x_gathered * d_Blockdense_Val[val_offset];
else sum += 0;

                r_x_gathered = __shfl_sync(0xffffffff, r_x,  xoff1 * BLOCK_SIZE + xoff1 + 12);
                val_offset += width;
if(val_offset<dense_size)
                sum += r_x_gathered * d_Blockdense_Val[val_offset];
else sum += 0;

                r_x_gathered = __shfl_sync(0xffffffff, r_x,  xoff1 * BLOCK_SIZE + xoff1 + 14);
                val_offset += width;
if(val_offset<dense_size)
                sum += r_x_gathered * d_Blockdense_Val[val_offset];
else sum += 0;


                // fuse sum at virtual_lane_id and it+16
                sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

return sum;

            }
