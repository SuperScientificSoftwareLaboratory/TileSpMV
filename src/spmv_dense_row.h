__inline__ __device__
int spmv_dense_row(int rbnum, int cbnum, int rowA, int colA, int blkj,int blki,int lane_id,int dense_size,int rowblkjstart,int collength,
int * d_denserowptr, MAT_VAL_TYPE *d_Blockdenserow_Val,char *d_denserowid,  MAT_VAL_TYPE *d_x,MAT_VAL_TYPE *s_y_warp,
int * s_dnsrow_offset_local, int x_offset ,MAT_VAL_TYPE sum) 
{
                int dnsrowoffset = s_dnsrow_offset_local[blkj - rowblkjstart]; 
                // load x needed by this block to register
                MAT_VAL_TYPE r_x = lane_id < collength ? d_x[x_offset + lane_id] : 0;//BLOCK_SIZE
                // copy the x to 2h of the 32 registers
                r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);
                
                int subwarp_id = lane_id / BLOCK_SIZE;
                int subwaprlane_id = (BLOCK_SIZE - 1) & lane_id; 

                int dnsrowptr = d_denserowptr[blkj];

                for (int ri = dnsrowptr + subwarp_id; ri < d_denserowptr[ blkj +1 ]; ri+=2)
                {
                    // get products
                    MAT_VAL_TYPE r_product = r_x * 
                                  d_Blockdenserow_Val[dnsrowoffset + (ri - dnsrowptr) * collength + subwaprlane_id];

                    // reduction sum on each half of the warp
                    if (lane_id < BLOCK_SIZE)
                    {
                        r_product += __shfl_down_sync(0x0000ffff, r_product, 8);
                        r_product += __shfl_down_sync(0x0000ffff, r_product, 4);
                        r_product += __shfl_down_sync(0x0000ffff, r_product, 2);
                        r_product += __shfl_down_sync(0x0000ffff, r_product, 1);
                    }
                    else
                    {
                        r_product += __shfl_down_sync(0xffff0000, r_product, 8);
                        r_product += __shfl_down_sync(0xffff0000, r_product, 4);
                        r_product += __shfl_down_sync(0xffff0000, r_product, 2);
                        r_product += __shfl_down_sync(0xffff0000, r_product, 1);
                    }

                    // copy the sum to smem
                    if (!subwaprlane_id)
                        s_y_warp[d_denserowid[ri]] += r_product;
                }

            }
