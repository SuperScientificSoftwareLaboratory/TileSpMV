#include "common.h"
#include "external/CSR5_cuda/anonymouslib_cuda.h"

// template <int SMEM_SIZE>
__global__ void stir_spmv_cuda_kernel_v5(int rbnum, int cbnum, int rowA, int colA,
                                         MAT_PTR_TYPE *d_rowblock_ptr,
                                         int *d_columnid,
                                         char *d_Format,
                                         int *d_blknnz,
                                         unsigned char *d_blknnznnz,
                                         unsigned char *d_csr_compressedIdx,
                                         MAT_VAL_TYPE *d_Blockcsr_Val,
                                         unsigned char *d_Blockcsr_Ptr,
                                         unsigned char *d_coo_Idx,
                                         MAT_VAL_TYPE *d_Blockcoo_Val,
                                         char *d_blkwidth,
                                         unsigned char *d_ell_compressedIdx,
                                         MAT_VAL_TYPE *d_Blockell_Val,
                                         unsigned char *d_hybIdx,
                                         MAT_VAL_TYPE *d_Blockhyb_Val,
                                         MAT_VAL_TYPE *d_Blockdense_Val,
                                         int *d_denserowptr,
                                         MAT_VAL_TYPE *d_Blockdenserow_Val,
                                         char *d_denserowid,
                                         int *d_densecolptr,
                                         MAT_VAL_TYPE *d_Blockdensecol_Val,
                                         char *d_densecolid,
                                         int *d_ptroffset1,
                                         int *d_ptroffset2,
                                         int rowblkblock,
                                         unsigned int *d_blkcoostylerowidx,
                                         int *d_blkcoostylerowidx_colstart,
                                         int *d_blkcoostylerowidx_colstop,
                                         MAT_VAL_TYPE *d_x,
                                         MAT_VAL_TYPE *d_y,
                                         int formatprofile,
                                         int *d_coodeferoffset,
                                         int *d_deferbuf_coooff,
                                         int *d_deferbuf_dxoff)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5; 
    __shared__ MAT_VAL_TYPE s_x[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ MAT_VAL_TYPE s_y[WARP_PER_BLOCK * BLOCK_SIZE];

    const int local_warp_id = threadIdx.x >> 5; 
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    MAT_VAL_TYPE *s_x_warp = &s_x[local_warp_id * BLOCK_SIZE];
    MAT_VAL_TYPE *s_y_warp = &s_y[local_warp_id * BLOCK_SIZE];

    __shared__ int s_deferbuf_coooff[WARP_PER_BLOCK * PREFETCH_SMEM_TH * COO_NNZ_TH];
    __shared__ int s_deferbuf_dxoff[WARP_PER_BLOCK * PREFETCH_SMEM_TH * COO_NNZ_TH];
    int *s_deferbuf_coooff_local = &s_deferbuf_coooff[local_warp_id * PREFETCH_SMEM_TH * COO_NNZ_TH];
    int *s_deferbuf_dxoff_local = &s_deferbuf_dxoff[local_warp_id * PREFETCH_SMEM_TH * COO_NNZ_TH];

    __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_ptroffset2[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ptroffset2_local = &s_ptroffset2[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ char s_Format[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    char *s_Format_local = &s_Format[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ unsigned char s_blknnznnz[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    unsigned char *s_blknnznnz_local = &s_blknnznnz[local_warp_id * PREFETCH_SMEM_TH];

    if (blki_blc < rowblkblock)
    {
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;

        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_rowblock_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_rowblock_ptr[blki + 1];

        if (lane_id < rowblkjstop - rowblkjstart)
        {
            s_columnid_local[lane_id] = d_columnid[rowblkjstart + lane_id];
            s_Format_local[lane_id] = d_Format[rowblkjstart + lane_id];
            s_blknnznnz_local[lane_id] = d_blknnznnz[rowblkjstart + lane_id];
            s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
            s_ptroffset2_local[lane_id] = d_ptroffset2[rowblkjstart + lane_id];
        }

        MAT_VAL_TYPE sum = 0;
        MAT_VAL_TYPE sumsum = 0;
        if (lane_id < BLOCK_SIZE)
            s_y_warp[lane_id] = 0;

        int coocnt = 0;
        int coodeferoffset = 0;

        // for each tile in the tile row
        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj++)
        {
            char subformat = s_Format_local[blkj - rowblkjstart];
            int colid = s_columnid_local[blkj - rowblkjstart];

            int collength = colid == cbnum - 1 ? colA - (cbnum - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int x_offset = colid * BLOCK_SIZE;

#if DEBUG_FORMATCOST
            if (formatprofile != 7) //
#endif
                switch (subformat)
                {
                // if CSR
                case 0:
#if DEBUG_FORMATCOST
                    if (formatprofile == 0 || formatprofile == -1)
#endif
                    {
                        sum = 0;
                        int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
                        int csrcount = s_ptroffset2_local[blkj - rowblkjstart];

                        if (lane_id < collength)
                            s_x_warp[lane_id] = d_x[x_offset + lane_id];

                        int ri = lane_id >> 1;               
                        int virtual_lane_id = lane_id & 0x1; 

                        int stop = ri == BLOCK_SIZE - 1 ? s_blknnznnz_local[blkj - rowblkjstart] : d_Blockcsr_Ptr[ri + 1 + csrcount];

                        for (int rj = d_Blockcsr_Ptr[csrcount + ri] + virtual_lane_id; rj < stop; rj += 2)
                        {
                            int csrcol = csroffset + rj;
                            unsigned char csridx = d_csr_compressedIdx[csrcol >> 1];
                            csrcol = csrcol % 2;
                            csrcol = csrcol == 0 ? (csridx & num_f) >> 4 : csridx & num_b;
                            sum += s_x_warp[csrcol] * d_Blockcsr_Val[csroffset + rj];
                        }

                        sum += __shfl_down_sync(0xffffffff, sum, 1);
                        sumsum += __shfl_down_sync(0xffffffff, sum, lane_id);
                    }

                    break;
                case 1:
                    // if COO
#if DEBUG_FORMATCOST
                    if (formatprofile == 1 || formatprofile == -1)
#endif
                    {
                        coocnt++;
                        // deferred
                        int blknnz = s_blknnznnz_local[blkj - rowblkjstart];
                        if (lane_id < blknnz)
                        {
                            s_deferbuf_coooff_local[coodeferoffset + lane_id] = s_ptroffset1_local[blkj - rowblkjstart] + lane_id;
                            s_deferbuf_dxoff_local[coodeferoffset + lane_id] = x_offset;
                        }
                        coodeferoffset += blknnz;
                    }

                    break;
                case 2:
                    // if ELL
#if DEBUG_FORMATCOST
                    if (formatprofile == 2 || formatprofile == -1)
#endif
                    {
                        sum = 0;
                        int ellwoffset = s_ptroffset1_local[blkj - rowblkjstart]; 

                        MAT_VAL_TYPE r_x = lane_id < collength ? d_x[x_offset + lane_id] : 0;

                        // produce all intermediate products
                        int elllen = d_blkwidth[blkj] * BLOCK_SIZE;
                        for (int rj = lane_id; rj < elllen; rj += WARP_SIZE)
                        {
                            int ellcol = ellwoffset + rj;
                            unsigned int ellidx = d_ell_compressedIdx[ellcol >> 1];
                            ellcol = ellcol % 2;
                            ellcol = ellcol == 0 ? (ellidx & num_f) >> 4 : ellidx & num_b;
                            MAT_VAL_TYPE r_x_gathered = lane_id < WARP_SIZE ? __shfl_sync(0x0000ffff, r_x, ellcol) : __shfl_sync(0xffff0000, r_x, ellcol);
                            sum += d_Blockell_Val[ellwoffset + rj] * r_x_gathered;
                        }
                        sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

                        sumsum += sum;
                    }

                    break;
                case 3:
                    // if HYB
#if DEBUG_FORMATCOST
                    if (formatprofile == 3 || formatprofile == -1)
#endif
                    {
                        // first do ELL (the above code can be called)

                        sum = 0;

                        int hybwoffset = s_ptroffset1_local[blkj - rowblkjstart];
                        int hybidxoffset = s_ptroffset2_local[blkj - rowblkjstart];

                        const MAT_VAL_TYPE r_x = lane_id < collength ? d_x[x_offset + lane_id] : 0;

                        int elllen = d_blkwidth[blkj] * BLOCK_SIZE;
                        for (int rj = lane_id; rj < elllen; rj += WARP_SIZE)
                        {
                            unsigned char hybidx = d_hybIdx[hybidxoffset + (rj >> 1)];
                            int hybcol = rj % 2 == 0 ? (hybidx & num_f) >> 4 : hybidx & num_b;

                            const MAT_VAL_TYPE r_x_gathered = __shfl_sync(0xffffffff, r_x, hybcol);
                            sum += d_Blockhyb_Val[hybwoffset + rj] * r_x_gathered;
                        }
                        sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

                        if (lane_id < BLOCK_SIZE)
                            sumsum += sum;

                        if (lane_id < BLOCK_SIZE)
                        {
                            s_x_warp[lane_id] = r_x;
                        }

                        hybidxoffset += elllen >> 1;
                        hybidxoffset += elllen % 2;

                        int nnzcoo = s_blknnznnz_local[blkj - rowblkjstart] - elllen;
                        for (int bnnzid = lane_id; bnnzid < nnzcoo; bnnzid += WARP_SIZE)
                        {
                            unsigned char hybidx = d_hybIdx[hybidxoffset + bnnzid];
                            unsigned char row = (hybidx & num_f) >> 4;
                            unsigned char col = hybidx & num_b;

                            MAT_VAL_TYPE r_x = d_Blockhyb_Val[elllen + hybwoffset + bnnzid] * s_x_warp[col];
                            atomicAdd(&s_y_warp[row], r_x);
                        }
                    }

                    break;
                case 4:
                    // if dense (or near dense stored as dense)
#if DEBUG_FORMATCOST
                    if (formatprofile == 4 || formatprofile == -1)
#endif
                    {
                        sum = 0;
                        int denseoffset = s_ptroffset1_local[blkj - rowblkjstart]; 

                        MAT_VAL_TYPE r_x = lane_id < BLOCK_SIZE ? d_x[x_offset + lane_id] : 0;
                        r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);
                        int xoff1 = lane_id >> 4; /// BLOCK_SIZE;

                        MAT_VAL_TYPE r_x_gathered;
                        int val_offset;
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1);
                        val_offset = denseoffset + lane_id;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 2);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 4);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 6);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 8);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 10);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 12);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 14);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

                        sumsum += sum;
                    }

                    break;
                case 5:
                    // if dense row (only work when the #nonzeros in the block is multiple of BLOCK_SIZE)
#if DEBUG_FORMATCOST
                    if (formatprofile == 5 || formatprofile == -1)
#endif
                    {
                        int dnsrowoffset = s_ptroffset1_local[blkj - rowblkjstart]; 
                        MAT_VAL_TYPE r_x = lane_id < BLOCK_SIZE ? d_x[x_offset + lane_id] : 0;
                        r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);

                        int subwarp_id = lane_id / BLOCK_SIZE;
                        int subwaprlane_id = (BLOCK_SIZE - 1) & lane_id; 

                        int dnsrowptr = d_denserowptr[blkj];
                        for (int ri = dnsrowptr + subwarp_id; ri < d_denserowptr[blkj + 1]; ri += 2)
                        {
                            // get products
                            MAT_VAL_TYPE r_product = r_x *
                                                     d_Blockdenserow_Val[dnsrowoffset + (ri - dnsrowptr) * collength + subwaprlane_id];

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

                    break;
                case 6:
                    // if dense col (only work when the #nonzeros in the block is multiple of BLOCK_SIZE)
#if DEBUG_FORMATCOST
                    if (formatprofile == 6 || formatprofile == -1)
#endif
                    {
                        sum = 0;
                        int dnscoloffset = s_ptroffset1_local[blkj - rowblkjstart]; // d_ptroffset1[blkj];

                        int colptrstart = d_densecolptr[blkj];
                        int dnswidth = d_densecolptr[blkj + 1] - colptrstart;

                        for (int glid = lane_id; glid < dnswidth * BLOCK_SIZE; glid += WARP_SIZE)
                        {
                            int rj = glid >> 4; // glid / BLOCK_SIZE;
                            int ri = glid % BLOCK_SIZE;
                            ri += dnscoloffset;
                            ri += rj * BLOCK_SIZE;
                            rj += colptrstart;
                            rj = d_densecolid[rj];
                            rj += x_offset;
                            sum += d_Blockdensecol_Val[ri] * d_x[rj];
                        }
                        sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

                        sumsum += sum;
                    }

                    break;
                }
        }

// deferred coo
#if DEBUG_FORMATCOST
        if (formatprofile == 1 || formatprofile == -1)
#endif

            if (coocnt > 0)
                d_coodeferoffset[blki_blc] = coodeferoffset;

        if (coocnt == rowblkjstop - rowblkjstart)
            d_coodeferoffset[blki_blc] = -coodeferoffset;

        for (int cooi = lane_id; cooi < coodeferoffset; cooi += WARP_SIZE)
        {
            if (coocnt > 0)
            {
                d_deferbuf_coooff[blki_blc * PREFETCH_SMEM_TH * COO_NNZ_TH + cooi] = s_deferbuf_coooff_local[cooi];
                d_deferbuf_dxoff[blki_blc * PREFETCH_SMEM_TH * COO_NNZ_TH + cooi] = s_deferbuf_dxoff_local[cooi];
            }
            int coooffset = s_deferbuf_coooff_local[cooi];
            unsigned char aidx = d_coo_Idx[coooffset];
            MAT_VAL_TYPE aval = d_Blockcoo_Val[coooffset];
            coooffset = s_deferbuf_dxoff_local[cooi];

            atomicAdd(&s_y_warp[(aidx & num_f) >> 4], aval * d_x[coooffset + (aidx & num_b)]);
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

__global__ void stir_spmv_cuda_kernel_v6(int tilem, int tilen, int rowA, int colA, int nnzA,
                                         MAT_PTR_TYPE *d_tile_ptr,
                                         int *d_tile_columnidx,
                                         char *d_Format,
                                         int *d_blknnz,
                                         unsigned char *d_blknnznnz,
                                         unsigned char *d_csr_compressedIdx,
                                         MAT_VAL_TYPE *d_Blockcsr_Val,
                                         unsigned char *d_Blockcsr_Ptr,
                                         unsigned char *d_coo_compressed_Idx,
                                         MAT_VAL_TYPE *d_Blockcoo_Val,
                                         char *d_tilewidth,
                                         unsigned char *d_ell_compressedIdx,
                                         MAT_VAL_TYPE *d_Blockell_Val,
                                         unsigned char *d_hybIdx,
                                         MAT_VAL_TYPE *d_Blockhyb_Val, MAT_VAL_TYPE *d_Blockdense_Val,
                                         int *d_dnsrowptr,
                                         MAT_VAL_TYPE *d_Blockdenserow_Val,
                                         char *d_denserowid,
                                         int *d_dnscolptr,
                                         MAT_VAL_TYPE *d_Blockdensecol_Val,
                                         char *d_densecolid,
                                         int *d_ptroffset1,
                                         int *d_ptroffset2,
                                         int rowblkblock,
                                         unsigned int *d_blkcoostylerowidx,
                                         int *d_blkcoostylerowidx_colstart,
                                         int *d_blkcoostylerowidx_colstop,
                                         MAT_VAL_TYPE *d_x,
                                         MAT_VAL_TYPE *d_y,
                                         int formatprofile,
                                         int *d_coodeferoffset,
                                         int *d_deferbuf_coooff,
                                         int *d_deferbuf_dxoff)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    __shared__ MAT_VAL_TYPE s_x[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ MAT_VAL_TYPE s_y[WARP_PER_BLOCK * BLOCK_SIZE];

    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    MAT_VAL_TYPE *s_x_warp = &s_x[local_warp_id * BLOCK_SIZE];
    MAT_VAL_TYPE *s_y_warp = &s_y[local_warp_id * BLOCK_SIZE];

    __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_ptroffset2[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ptroffset2_local = &s_ptroffset2[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ char s_Format[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    char *s_Format_local = &s_Format[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ unsigned char s_blknnznnz[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    unsigned char *s_blknnznnz_local = &s_blknnznnz[local_warp_id * PREFETCH_SMEM_TH];

    if (blki_blc < rowblkblock)
    {
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;

        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;

        MAT_VAL_TYPE sum = 0;
        MAT_VAL_TYPE sumsum = 0;
        if (lane_id < BLOCK_SIZE)
            s_y_warp[lane_id] = 0;
        // coo speculative
        int coodeferoffset_global = d_coodeferoffset[blki_blc];
        int cooall = coodeferoffset_global < 0 ? 1 : 0;
        coodeferoffset_global = coodeferoffset_global > 0 ? coodeferoffset_global : -coodeferoffset_global;

        if (coodeferoffset_global > 0)
        {
            for (int cooi = lane_id; cooi < coodeferoffset_global; cooi += WARP_SIZE)
            {
                int coooffset = d_deferbuf_coooff[blki_blc * PREFETCH_SMEM_TH * COO_NNZ_TH + cooi];
                // int cooxoffset = s_deferbuf_dxoff_local[cooi];

                unsigned char aidx = d_coo_compressed_Idx[coooffset];
                // unsigned char row = (aidx & num_f) >> 4;
                // unsigned char col =  aidx & num_b;
                MAT_VAL_TYPE aval = d_Blockcoo_Val[coooffset];
                // MAT_VAL_TYPE r_x = aval * d_x[x_offset + aidx & num_b];
                coooffset = d_deferbuf_dxoff[blki_blc * PREFETCH_SMEM_TH * COO_NNZ_TH + cooi];

                atomicAdd(&s_y_warp[(aidx & num_f) >> 4], aval * d_x[coooffset + (aidx & num_b)]);
            }

            if (lane_id < BLOCK_SIZE)
            {
                sumsum = s_y_warp[lane_id];
                s_y_warp[lane_id] = 0;
            }
        }
        if (cooall)
        {
            // save sum to d_y
            if (lane_id < BLOCK_SIZE && sumsum != 0)
            {
                if (signbit)
                    atomicAdd(&d_y[blki * BLOCK_SIZE + lane_id], sumsum);
                else
                    d_y[blki * BLOCK_SIZE + lane_id] = sumsum;
            }
            return;
        }

        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_tile_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_tile_ptr[blki + 1];

        if (lane_id < rowblkjstop - rowblkjstart)
        {
            s_columnid_local[lane_id] = d_tile_columnidx[rowblkjstart + lane_id];
            s_Format_local[lane_id] = d_Format[rowblkjstart + lane_id];
            s_blknnznnz_local[lane_id] = d_blknnznnz[rowblkjstart + lane_id];
            s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
            s_ptroffset2_local[lane_id] = d_ptroffset2[rowblkjstart + lane_id];
        }

        int coodeferoffset = 0;

        // for each block in the block row
        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj++)
        {
            char subformat = s_Format_local[blkj - rowblkjstart];
            int colid = s_columnid_local[blkj - rowblkjstart];

            int collength = colid == tilen - 1 ? colA - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int x_offset = colid * BLOCK_SIZE;

#if DEBUG_FORMATCOST
            if (formatprofile != 7) //
#endif
                switch (subformat)
                {
                // if CSR
                case 0:
#if DEBUG_FORMATCOST
                    if (formatprofile == 0 || formatprofile == -1)
#endif
                    {
                        sum = 0;
                        int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
                        int csrcount = s_ptroffset2_local[blkj - rowblkjstart];

                        if (lane_id < collength)
                            s_x_warp[lane_id] = d_x[x_offset + lane_id];

                        int ri = lane_id >> 1;
                        int virtual_lane_id = lane_id & 0x1;

                        int stop = ri == BLOCK_SIZE - 1 ? s_blknnznnz_local[blkj - rowblkjstart] : d_Blockcsr_Ptr[ri + 1 + csrcount];

                        for (int rj = d_Blockcsr_Ptr[csrcount + ri] + virtual_lane_id; rj < stop; rj += 2)
                        {
                            int csrcol = csroffset + rj;
                            unsigned char csridx = d_csr_compressedIdx[csrcol >> 1];
                            csrcol = csrcol % 2;
                            csrcol = csrcol == 0 ? (csridx & num_f) >> 4 : csridx & num_b;
                            sum += s_x_warp[csrcol] * d_Blockcsr_Val[csroffset + rj];
                        }

                        sum += __shfl_down_sync(0xffffffff, sum, 1);
                        sumsum += __shfl_down_sync(0xffffffff, sum, lane_id);
                    }

                    break;
                case 1:
                    // if COO
#if DEBUG_FORMATCOST
                    if (formatprofile == 1 || formatprofile == -1)
#endif
                        // if (nnzA < 1800000)
                        // {
                        //     int coooffset = d_ptroffset1[blkj] + lane_id;

                        //     int blknnz = s_blknnznnz_local[blkj - rowblkjstart];
                        //     unsigned char aidx = lane_id < blknnz ? d_coo_compressed_Idx[coooffset] : 0;
                        //     MAT_VAL_TYPE aval = lane_id < blknnz ? d_Blockcoo_Val[coooffset] : 0;
                        //     if (lane_id < blknnz)
                        //         atomicAdd(&s_y_warp[(aidx & num_f) >> 4], aval * d_x[x_offset + (aidx & num_b)]);
                        // }

                        break;
                case 2:
                    // if ELL
#if DEBUG_FORMATCOST
                    if (formatprofile == 2 || formatprofile == -1)
#endif
                    {
                        sum = 0;
                        int ellwoffset = s_ptroffset1_local[blkj - rowblkjstart];

                        MAT_VAL_TYPE r_x = lane_id < collength ? d_x[x_offset + lane_id] : 0;

                        int elllen = d_tilewidth[blkj] * BLOCK_SIZE;
                        for (int rj = lane_id; rj < elllen; rj += WARP_SIZE)
                        {
                            int ellcol = ellwoffset + rj;
                            unsigned int ellidx = d_ell_compressedIdx[ellcol >> 1];
                            ellcol = ellcol % 2;
                            ellcol = ellcol == 0 ? (ellidx & num_f) >> 4 : ellidx & num_b;
                            MAT_VAL_TYPE r_x_gathered = lane_id < WARP_SIZE ? __shfl_sync(0x0000ffff, r_x, ellcol) : __shfl_sync(0xffff0000, r_x, ellcol);
                            sum += d_Blockell_Val[ellwoffset + rj] * r_x_gathered;
                        }
                        sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

                        sumsum += sum;
                    }

                    break;
                case 3:
                    // if ELL

#if DEBUG_FORMATCOST
                    if (formatprofile == 3 || formatprofile == -1)
#endif
                    {
                        // first do ELL (the above code can be called)
                        sum = 0;

                        int hybwoffset = s_ptroffset1_local[blkj - rowblkjstart];
                        int hybidxoffset = s_ptroffset2_local[blkj - rowblkjstart];

                        const MAT_VAL_TYPE r_x = lane_id < collength ? d_x[x_offset + lane_id] : 0;

                        int elllen = d_tilewidth[blkj] * BLOCK_SIZE;
                        for (int rj = lane_id; rj < elllen; rj += WARP_SIZE)
                        {
                            unsigned char hybidx = d_hybIdx[hybidxoffset + (rj >> 1)];
                            int hybcol = rj % 2 == 0 ? (hybidx & num_f) >> 4 : hybidx & num_b;

                            const MAT_VAL_TYPE r_x_gathered = __shfl_sync(0xffffffff, r_x, hybcol);
                            sum += d_Blockhyb_Val[hybwoffset + rj] * r_x_gathered;
                        }
                        sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

                        if (lane_id < BLOCK_SIZE)
                            sumsum += sum;

                        // // then do COO (the above code can be called)
                        // if (lane_id < BLOCK_SIZE)
                        // {
                        //     // s_y_warp[lane_id] = 0;
                        //     s_x_warp[lane_id] = r_x;
                        // }

                        // hybidxoffset += elllen >> 1; /// 2;
                        // hybidxoffset += elllen % 2;

                        // // int blknnz = s_blknnznnz_local[blkj - rowblkjstart];
                        // // int nnzcoo = blknnz - elllen;

                        // int nnzcoo = s_blknnznnz_local[blkj - rowblkjstart] - elllen;
                        // for (int bnnzid = lane_id; bnnzid < nnzcoo; bnnzid += WARP_SIZE)
                        // {
                        //     unsigned char hybidx = d_hybIdx[hybidxoffset + bnnzid];
                        //     unsigned char row = (hybidx & num_f) >> 4;
                        //     unsigned char col = hybidx & num_b;

                        //     MAT_VAL_TYPE r_x = d_Blockhyb_Val[elllen + hybwoffset + bnnzid] * s_x_warp[col];
                        //     atomicAdd(&s_y_warp[row], r_x);
                        // }

                        // if (lane_id < BLOCK_SIZE)
                        //     sumsum += s_y_warp[lane_id];
                    }

                    break;
                case 4:
                    // if dense (or near dense stored as dense)
#if DEBUG_FORMATCOST
                    if (formatprofile == 4 || formatprofile == -1)
#endif
                    {
                        sum = 0;
                        int denseoffset = s_ptroffset1_local[blkj - rowblkjstart];

                        MAT_VAL_TYPE r_x = lane_id < BLOCK_SIZE ? d_x[x_offset + lane_id] : 0;

                        r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);
                        int xoff1 = lane_id >> 4;

                        MAT_VAL_TYPE r_x_gathered;
                        int val_offset;
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1);
                        val_offset = denseoffset + lane_id;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 2);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 4);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 6);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 8);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 10);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 12);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];
                        r_x_gathered = __shfl_sync(0xffffffff, r_x, xoff1 * BLOCK_SIZE + xoff1 + 14);
                        val_offset += WARP_SIZE;
                        sum += r_x_gathered * d_Blockdense_Val[val_offset];

                        sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

                        sumsum += sum;
                    }

                    break;
                case 5:
                    // if dense row (only work when the #nonzeros in the block is multiple of BLOCK_SIZE)
#if DEBUG_FORMATCOST
                    if (formatprofile == 5 || formatprofile == -1)
#endif
                    {
                        int dnsrowoffset = s_ptroffset1_local[blkj - rowblkjstart];
                        MAT_VAL_TYPE r_x = lane_id < BLOCK_SIZE ? d_x[x_offset + lane_id] : 0;
                        r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);

                        int subwarp_id = lane_id / BLOCK_SIZE;
                        int subwaprlane_id = (BLOCK_SIZE - 1) & lane_id;

                        int dnsrowptr = d_dnsrowptr[blkj];
                        for (int ri = dnsrowptr + subwarp_id; ri < d_dnsrowptr[blkj + 1]; ri += 2)
                        {
                            MAT_VAL_TYPE r_product = r_x *
                                                     d_Blockdenserow_Val[dnsrowoffset + (ri - dnsrowptr) * collength + subwaprlane_id];

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

                            if (!subwaprlane_id)
                                s_y_warp[d_denserowid[ri]] += r_product;
                        }
                    }

                    break;
                case 6:
                    // if dense col (only work when the #nonzeros in the block is multiple of BLOCK_SIZE)
#if DEBUG_FORMATCOST
                    if (formatprofile == 6 || formatprofile == -1)
#endif
                    {
                        sum = 0;
                        int dnscoloffset = s_ptroffset1_local[blkj - rowblkjstart];
                        int colptrstart = d_dnscolptr[blkj];
                        int dnswidth = d_dnscolptr[blkj + 1] - colptrstart;

                        for (int glid = lane_id; glid < dnswidth * BLOCK_SIZE; glid += WARP_SIZE)
                        {
                            int rj = glid >> 4;
                            int ri = glid % BLOCK_SIZE;
                            ri += dnscoloffset;
                            ri += rj * BLOCK_SIZE;
                            rj += colptrstart;
                            rj = d_densecolid[rj];
                            rj += x_offset;
                            sum += d_Blockdensecol_Val[ri] * d_x[rj];
                        }
                        sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);

                        sumsum += sum;
                    }

                    break;
                }
        }
        if (lane_id < BLOCK_SIZE)
            sumsum += s_y_warp[lane_id];

        if (lane_id < BLOCK_SIZE && sumsum != 0)
        {
            if (signbit)
                atomicAdd(&d_y[blki * BLOCK_SIZE + lane_id], sumsum);
            else
                d_y[blki * BLOCK_SIZE + lane_id] = sumsum;
        }
    }
}

void call_tilespmv_cuda(char *filename,
                        Tile_matrix *matrix,
                        int *ptroffset1,
                        int *ptroffset2,
                        int rowblkblock,
                        unsigned int *blkcoostylerowidx,
                        int *blkcoostylerowidx_colstart,
                        int *blkcoostylerowidx_colstop,
                        int rowA, int colA, MAT_PTR_TYPE nnzA,
                        MAT_PTR_TYPE *csrRowPtrA,
                        int *csrColIdxA,
                        MAT_VAL_TYPE *csrValA,
                        MAT_VAL_TYPE alpha,
                        MAT_VAL_TYPE *x,
                        MAT_VAL_TYPE *y,
                        MAT_VAL_TYPE *y_golden)
{
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    int *blknnz = matrix->blknnz;
    unsigned char *blknnznnz = matrix->blknnznnz;
    char *tilewidth = matrix->tilewidth;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;
    int *coo_offset = matrix->coo_offset;
    int *ell_offset = matrix->ell_offset;
    int *hyb_offset = matrix->hyb_offset;
    int *hyb_coocount = matrix->hyb_coocount;
    int *dns_offset = matrix->dns_offset;
    int *dnsrowptr = matrix->dnsrowptr;
    int *dnsrow_offset = matrix->dnsrow_offset;
    int *dnscolptr = matrix->dnscolptr;
    int *dnscol_offset = matrix->dnscol_offset;
    int *new_coocount = matrix->new_coocount;
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
    unsigned char *csr_compressedIdx = matrix->csr_compressedIdx;
    unsigned char *Blockcsr_Ptr = matrix->Blockcsr_Ptr;
    MAT_VAL_TYPE *Blockcoo_Val = matrix->Blockcoo_Val;
    unsigned char *coo_compressed_Idx = matrix->coo_compressed_Idx;
    MAT_VAL_TYPE *Blockell_Val = matrix->Blockell_Val;
    unsigned char *ell_compressedIdx = matrix->ell_compressedIdx;
    MAT_VAL_TYPE *Blockhyb_Val = matrix->Blockhyb_Val;
    unsigned char *hybIdx = matrix->hybIdx;
    MAT_VAL_TYPE *Blockdense_Val = matrix->Blockdense_Val;
    MAT_VAL_TYPE *Blockdenserow_Val = matrix->Blockdenserow_Val;
    char *denserowid = matrix->denserowid;
    MAT_VAL_TYPE *Blockdensecol_Val = matrix->Blockdensecol_Val;
    char *densecolid = matrix->densecolid;
    int csrsize = matrix->csrsize;
    int csrptrlen = matrix->csrptrlen;
    int coosize = matrix->coosize;
    int ellsize = matrix->ellsize;
    int hybcoosize = matrix->hybcoosize;
    int hybellsize = matrix->hybellsize;
    int dense_size = matrix->dnssize;
    int denserow_size = matrix->dnsrowsize;
    int densecol_size = matrix->dnscolsize;
    int tilenum = matrix->tilenum;
    int coototal = matrix->coototal;
    MAT_PTR_TYPE *deferredcoo_ptr = matrix->deferredcoo_ptr;
    int *deferredcoo_colidx = matrix->deferredcoo_colidx;
    MAT_VAL_TYPE *deferredcoo_val = matrix->deferredcoo_val;

    int csr_csize = csrsize % 2 == 0 ? csrsize / 2 : csrsize / 2 + 1;
    int ell_csize = ellsize % 2 == 0 ? ellsize / 2 : ellsize / 2 + 1;
    int hyb_size = hybellsize % 2 == 0 ? hybellsize / 2 : (hybellsize / 2) + 1;

    // tile matrix

    MAT_PTR_TYPE *d_tile_ptr;
    int *d_tile_columnidx;
    char *d_Format;
    int *d_blknnz;
    unsigned char *d_blknnznnz;

    cudaMalloc((void **)&d_tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void **)&d_tile_columnidx, tilenum * sizeof(int));
    cudaMalloc((void **)&d_Format, tilenum * sizeof(char));
    cudaMalloc((void **)&d_blknnz, (tilenum + 1) * sizeof(int));
    cudaMalloc((void **)&d_blknnznnz, (tilenum + 1) * sizeof(unsigned char));

    cudaMemcpy(d_tile_ptr, tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_columnidx, tile_columnidx, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Format, Format, tilenum * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blknnz, blknnz, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blknnznnz, blknnznnz, (tilenum + 1) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // CSR
    unsigned char *d_csr_compressedIdx = (unsigned char *)malloc((csr_csize) * sizeof(unsigned char));
    MAT_VAL_TYPE *d_Blockcsr_Val;
    unsigned char *d_Blockcsr_Ptr;

    cudaMalloc((void **)&d_csr_compressedIdx, (csr_csize) * sizeof(unsigned char));
    cudaMalloc((void **)&d_Blockcsr_Val, (csrsize) * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_Blockcsr_Ptr, (csrptrlen) * sizeof(unsigned char));

    cudaMemcpy(d_csr_compressedIdx, csr_compressedIdx, (csr_csize) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcsr_Val, Blockcsr_Val, (csrsize) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcsr_Ptr, Blockcsr_Ptr, (csrptrlen) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // COO
    unsigned char *d_coo_compressed_Idx;
    MAT_VAL_TYPE *d_Blockcoo_Val;

    cudaMalloc((void **)&d_coo_compressed_Idx, (coosize) * sizeof(unsigned char));
    cudaMalloc((void **)&d_Blockcoo_Val, (coosize) * sizeof(MAT_VAL_TYPE));

    cudaMemcpy(d_coo_compressed_Idx, coo_compressed_Idx, (coosize) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcoo_Val, Blockcoo_Val, (coosize) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    // ELL
    unsigned char *d_ell_compressedIdx;
    MAT_VAL_TYPE *d_Blockell_Val;

    cudaMalloc((void **)&d_ell_compressedIdx, (ell_csize) * sizeof(unsigned char));
    cudaMalloc((void **)&d_Blockell_Val, (ellsize) * sizeof(MAT_VAL_TYPE));

    cudaMemcpy(d_ell_compressedIdx, ell_compressedIdx, (ell_csize) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockell_Val, Blockell_Val, (ellsize) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    // HYB
    unsigned char *d_hybIdx;
    char *d_tilewidth;
    MAT_VAL_TYPE *d_Blockhyb_Val;

    cudaMalloc((void **)&d_hybIdx, (hyb_size + hybcoosize) * sizeof(unsigned char));
    cudaMalloc((void **)&d_tilewidth, tilenum * sizeof(char));
    cudaMalloc((void **)&d_Blockhyb_Val, (hybellsize + hybcoosize) * sizeof(MAT_VAL_TYPE));

    cudaMemcpy(d_hybIdx, hybIdx, (hyb_size + hybcoosize) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tilewidth, tilewidth, tilenum * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockhyb_Val, Blockhyb_Val, (hybellsize + hybcoosize) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    // dense
    MAT_VAL_TYPE *d_Blockdense_Val;

    cudaMalloc((void **)&d_Blockdense_Val, (dense_size) * sizeof(MAT_VAL_TYPE));

    cudaMemcpy(d_Blockdense_Val, Blockdense_Val, (dense_size) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    // denserow
    int *d_dnsrowptr;
    MAT_VAL_TYPE *d_Blockdenserow_Val;
    char *d_denserowid;

    cudaMalloc((void **)&d_dnsrowptr, (tilenum + 1) * sizeof(int));
    cudaMalloc((void **)&d_Blockdenserow_Val, (denserow_size) * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_denserowid, dnsrowptr[tilenum] * sizeof(char));

    cudaMemcpy(d_dnsrowptr, dnsrowptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockdenserow_Val, Blockdenserow_Val, (denserow_size) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_denserowid, denserowid, dnsrowptr[tilenum] * sizeof(char), cudaMemcpyHostToDevice);

    // dense column
    int *d_dnscolptr;
    MAT_VAL_TYPE *d_Blockdensecol_Val;
    char *d_densecolid;

    cudaMalloc((void **)&d_dnscolptr, (tilenum + 1) * sizeof(int));
    cudaMalloc((void **)&d_Blockdensecol_Val, (densecol_size) * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_densecolid, dnscolptr[tilenum] * sizeof(char));

    cudaMemcpy(d_dnscolptr, dnscolptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockdensecol_Val, Blockdensecol_Val, (densecol_size) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_densecolid, densecolid, dnscolptr[tilenum] * sizeof(char), cudaMemcpyHostToDevice);

    unsigned int *d_blkcoostylerowidx;
    int *d_blkcoostylerowidx_colstart;
    int *d_blkcoostylerowidx_colstop;

    cudaMalloc((void **)&d_blkcoostylerowidx, rowblkblock * sizeof(unsigned int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstart, rowblkblock * sizeof(int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstop, rowblkblock * sizeof(int));

    cudaMemcpy(d_blkcoostylerowidx, blkcoostylerowidx, rowblkblock * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstart, blkcoostylerowidx_colstart, rowblkblock * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstop, blkcoostylerowidx_colstop, rowblkblock * sizeof(int), cudaMemcpyHostToDevice);

    int *d_ptroffset1;
    int *d_ptroffset2;

    cudaMalloc((void **)&d_ptroffset1, tilenum * sizeof(int));
    cudaMalloc((void **)&d_ptroffset2, tilenum * sizeof(int));
    cudaMemcpy(d_ptroffset1, ptroffset1, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptroffset2, ptroffset2, tilenum * sizeof(int), cudaMemcpyHostToDevice);

    // deferred coo
    MAT_PTR_TYPE *d_deferredcoo_ptr;
    int *d_deferredcoo_colidx;
    MAT_VAL_TYPE *d_deferredcoo_val;

    cudaMalloc((void **)&d_deferredcoo_ptr, (rowA + 1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void **)&d_deferredcoo_colidx, (coototal) * sizeof(int));
    cudaMalloc((void **)&d_deferredcoo_val, (coototal) * sizeof(MAT_VAL_TYPE));

    cudaMemcpy(d_deferredcoo_ptr, deferredcoo_ptr, (rowA + 1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deferredcoo_colidx, deferredcoo_colidx, coototal * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deferredcoo_val, deferredcoo_val, coototal * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    // x and y
    MAT_VAL_TYPE *d_x;
    MAT_VAL_TYPE *d_y;

    cudaMalloc((void **)&d_x, colA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_y, rowA * sizeof(MAT_VAL_TYPE));

    cudaMemcpy(d_x, x, colA * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    // CSR5

    int err = 0;
    cudaError_t err_cuda = cudaSuccess;
    double gflop = getFLOP<int>(nnzA);
    anonymouslibHandle<int, unsigned int, MAT_VAL_TYPE> A(rowA, colA);
    if (coototal != 0)
    {

        err = A.inputCSR(coototal, d_deferredcoo_ptr, d_deferredcoo_colidx, d_deferredcoo_val);

        err = A.setX(d_x);

        A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);

        // warmup device
        A.warmup();

        anonymouslib_timer asCSR5_timer;
        asCSR5_timer.start();

        err = A.asCSR5();
        double csr2csr5 = asCSR5_timer.stop();
    }
    int *d_coodeferoffset;
    int *d_deferbuf_coooff;
    int *d_deferbuf_dxoff;

    cudaMalloc((void **)&d_coodeferoffset, rowblkblock * sizeof(int));
    cudaMemset(d_coodeferoffset, 0, rowblkblock * sizeof(int));

    cudaMalloc((void **)&d_deferbuf_coooff, rowblkblock * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMemset(d_deferbuf_coooff, 0, rowblkblock * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMalloc((void **)&d_deferbuf_dxoff, rowblkblock * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMemset(d_deferbuf_dxoff, 0, rowblkblock * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));

    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));

    stir_spmv_cuda_kernel_v5<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA,
                                                          d_tile_ptr, d_tile_columnidx, d_Format, d_blknnz, d_blknnznnz,
                                                          d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                          d_coo_compressed_Idx, d_Blockcoo_Val,
                                                          d_tilewidth, d_ell_compressedIdx, d_Blockell_Val,
                                                          d_hybIdx, d_Blockhyb_Val,
                                                          d_Blockdense_Val,
                                                          d_dnsrowptr, d_Blockdenserow_Val, d_denserowid,
                                                          d_dnscolptr, d_Blockdensecol_Val, d_densecolid,
                                                          d_ptroffset1, d_ptroffset2,
                                                          rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                          d_x, d_y, 7, d_coodeferoffset, d_deferbuf_coooff, d_deferbuf_dxoff);

    // warm up by running 200 times
    for (int i = 0; i < WARMUP_NUM; i++)
    {
        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
        int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
        cudaMemset(d_y, 0, rowA * sizeof(MAT_VAL_TYPE));

        stir_spmv_cuda_kernel_v6<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA, nnzA,
                                                              d_tile_ptr, d_tile_columnidx, d_Format, d_blknnz, d_blknnznnz,
                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                              d_coo_compressed_Idx, d_Blockcoo_Val,
                                                              d_tilewidth, d_ell_compressedIdx, d_Blockell_Val,
                                                              d_hybIdx, d_Blockhyb_Val,
                                                              d_Blockdense_Val,
                                                              d_dnsrowptr, d_Blockdenserow_Val, d_denserowid,
                                                              d_dnscolptr, d_Blockdensecol_Val, d_densecolid,
                                                              d_ptroffset1, d_ptroffset2,
                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                              d_x, d_y, 7, d_coodeferoffset, d_deferbuf_coooff, d_deferbuf_dxoff);

        // call csr5
        if (coototal != 0)
            err = A.spmv(alpha, d_y);
    }
    cudaDeviceSynchronize();

    for (int fi = 0; fi < 4; fi++)
    {
        for (int i = 0; i < BENCH_REPEAT; i++)
        {
            int num_threads = WARP_PER_BLOCK * WARP_SIZE;
            int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
            cudaMemset(d_y, 0, rowA * sizeof(MAT_VAL_TYPE));

            stir_spmv_cuda_kernel_v6<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA, nnzA,
                                                                  d_tile_ptr, d_tile_columnidx, d_Format, d_blknnz, d_blknnznnz,
                                                                  d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                  d_coo_compressed_Idx, d_Blockcoo_Val,
                                                                  d_tilewidth, d_ell_compressedIdx, d_Blockell_Val,
                                                                  d_hybIdx, d_Blockhyb_Val,
                                                                  d_Blockdense_Val,
                                                                  d_dnsrowptr, d_Blockdenserow_Val, d_denserowid,
                                                                  d_dnscolptr, d_Blockdensecol_Val, d_densecolid,
                                                                  d_ptroffset1, d_ptroffset2,
                                                                  rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                  d_x, d_y, 7, d_coodeferoffset, d_deferbuf_coooff, d_deferbuf_dxoff);
            if (coototal != 0 && nnzA >= 1800000)
                err = A.spmv(alpha, d_y);
            cudaDeviceSynchronize();
        }
    }

    timeval t1, t2;
    double time_cuda_spmv_base = 0;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
        int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
        cudaMemset(d_y, 0, rowA * sizeof(MAT_VAL_TYPE));

        gettimeofday(&t1, NULL);
        stir_spmv_cuda_kernel_v6<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA, nnzA,
                                                              d_tile_ptr, d_tile_columnidx, d_Format, d_blknnz, d_blknnznnz,
                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                              d_coo_compressed_Idx, d_Blockcoo_Val,
                                                              d_tilewidth, d_ell_compressedIdx, d_Blockell_Val,
                                                              d_hybIdx, d_Blockhyb_Val,
                                                              d_Blockdense_Val,
                                                              d_dnsrowptr, d_Blockdenserow_Val, d_denserowid,
                                                              d_dnscolptr, d_Blockdensecol_Val, d_densecolid,
                                                              d_ptroffset1, d_ptroffset2,
                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                              d_x, d_y, 7, d_coodeferoffset, d_deferbuf_coooff, d_deferbuf_dxoff);
        // if (coototal != 0 && nnzA >= 1800000)
        //     err = A.spmv(alpha, d_y);
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time_cuda_spmv_base += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }
    time_cuda_spmv_base /= BENCH_REPEAT;
    double gflops = 2 * (double)nnzA * 1.0e-6 / time_cuda_spmv_base;
    printf("  CUDA SpMV runtime %4.2f ms, %4.2f GFlops\n\n", time_cuda_spmv_base, gflops);

    // write results to text (scv) file
    FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%i,%i,%i,%f,%f\n",
            filename, rowA, colA, nnzA, time_cuda_spmv_base, gflops);
    fclose(fout);

    cudaMemcpy(y, d_y, rowA * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);

    // matrix
    cudaFree(d_tile_ptr);
    cudaFree(d_tile_columnidx);
    cudaFree(d_Format);
    cudaFree(d_blknnz);
    // CSR
    cudaFree(d_csr_compressedIdx);
    cudaFree(d_Blockcsr_Val);
    cudaFree(d_Blockcsr_Ptr);
    // COO
    cudaFree(d_coo_compressed_Idx);
    cudaFree(d_Blockcoo_Val);
    // ELL
    cudaFree(d_tilewidth);
    cudaFree(d_ell_compressedIdx);
    cudaFree(d_Blockell_Val);
    // HYB
    cudaFree(d_hybIdx);
    cudaFree(d_Blockhyb_Val);
    // dense
    cudaFree(d_Blockdense_Val);
    // denserow
    cudaFree(d_dnsrowptr);
    cudaFree(d_Blockdenserow_Val);
    cudaFree(d_denserowid);
    // densecol
    cudaFree(d_dnscolptr);
    cudaFree(d_Blockdensecol_Val);
    cudaFree(d_densecolid);
}
