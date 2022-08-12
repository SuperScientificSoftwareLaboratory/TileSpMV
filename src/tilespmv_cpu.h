#include "common.h"

void tilespmv_cpu(Tile_matrix *matrix,
                  int *ptroffset1,
                  int *ptroffset2,
                  int *rowblkblock,
                  unsigned int **blkcoostylerowidx,
                  int **blkcoostylerowidx_colstart,
                  int **blkcoostylerowidx_colstop,
                  int rowA, int colA, MAT_PTR_TYPE nnzA,
                  MAT_PTR_TYPE *csrRowPtrA,
                  int *csrColIdxA,
                  MAT_VAL_TYPE *csrValA,
                  MAT_VAL_TYPE *x,
                  MAT_VAL_TYPE *y,
                  MAT_VAL_TYPE *y_golden

)

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

    int ellwoffset = 0;
    int hybwoffset = 0;
    int hybidxoffset = 0;
    int denseoffset = 0;
    int coooffset = 0;
    int csroffset = 0;
    int csrcount = 0;
    int dnsrowoffset = 0;
    int dnscoloffset = 0;

    // balance analysis
    int rowblkblock_tmp = 0;
    for (int blki = 0; blki < tilem; blki++)
    {
        int balancenumblk = tile_ptr[blki + 1] - tile_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH)
            rowblkblock_tmp++;
        else
        {
            rowblkblock_tmp += ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH);
        }
    }
    *rowblkblock = rowblkblock_tmp;

    *blkcoostylerowidx = (unsigned int *)malloc(sizeof(unsigned int) * *rowblkblock);
    unsigned int *blkcoostylerowidx_tmp = *blkcoostylerowidx;
    memset(blkcoostylerowidx_tmp, 0, sizeof(unsigned int) * *rowblkblock);

    *blkcoostylerowidx_colstart = (int *)malloc(sizeof(int) * *rowblkblock);
    int *blkcoostylerowidx_colstart_tmp = *blkcoostylerowidx_colstart;
    memset(blkcoostylerowidx_colstart_tmp, 0, sizeof(int) * *rowblkblock);
    *blkcoostylerowidx_colstop = (int *)malloc(sizeof(int) * *rowblkblock);
    int *blkcoostylerowidx_colstop_tmp = *blkcoostylerowidx_colstop;
    memset(blkcoostylerowidx_colstop_tmp, 0, sizeof(int) * *rowblkblock);

    int rowblkblockcnt = 0;
    for (int blki = 0; blki < tilem; blki++)
    {
        int balancenumblk = tile_ptr[blki + 1] - tile_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH)
        {
            blkcoostylerowidx_tmp[rowblkblockcnt] = blki;
            rowblkblockcnt++;
        }
        else
        {
            int numblklocal = ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH);
            int lenblklocal = ceil((double)balancenumblk / (double)numblklocal);
            for (int iii = 0; iii < numblklocal; iii++)
            {
                blkcoostylerowidx_tmp[rowblkblockcnt] = blki | 0x80000000; // can generate -0
                blkcoostylerowidx_colstart_tmp[rowblkblockcnt] = tile_ptr[blki] + iii * lenblklocal;
                if (iii == numblklocal - 1)
                    blkcoostylerowidx_colstop_tmp[rowblkblockcnt] = tile_ptr[blki] + balancenumblk;
                else
                    blkcoostylerowidx_colstop_tmp[rowblkblockcnt] = tile_ptr[blki] + (iii + 1) * lenblklocal;

                rowblkblockcnt++;
            }
        }
    }

    int *formathistogram = (int *)malloc(7 * sizeof(int));
    memset(formathistogram, 0, 7 * sizeof(int));

    int coototalcnt = 0;

    for (int blki = 0; blki < tilem; blki++)
    {
        int rowlength = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int ri = 0; ri < BLOCK_SIZE; ri++)
        {
            y[blki * BLOCK_SIZE + ri] = 0;
        }
        for (int blkj = tile_ptr[blki]; blkj < tile_ptr[blki + 1]; blkj++)
        {
            int collength = tile_columnidx[blkj] == tilen - 1 ? colA - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int x_offset = tile_columnidx[blkj] * BLOCK_SIZE;
            formathistogram[Format[blkj]]++;
            char format = Format[blkj];
            switch (format)
            {
            case 0:
            {
                ptroffset1[blkj] = csroffset;
                ptroffset2[blkj] = csrcount;
                for (int ri = 0; ri < rowlength; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    int stop = ri == rowlength - 1 ? (blknnz[blkj + 1] - blknnz[blkj]) : Blockcsr_Ptr[ri + 1 + csrcount];
                    for (int rj = Blockcsr_Ptr[csrcount + ri]; rj < stop; rj++)
                    {
                        int csrcol = (csroffset + rj) % 2 == 0 ? (csr_compressedIdx[(csroffset + rj) / 2] & num_f) >> 4 : csr_compressedIdx[(csroffset + rj) / 2] & num_b;
                        sum += x[x_offset + csrcol] * Blockcsr_Val[csroffset + rj];
                    }
                    y[blki * BLOCK_SIZE + ri] += sum;
                }
                csroffset += blknnz[blkj + 1] - blknnz[blkj];
                csrcount += rowlength;
                break;
            }
            case 1:
            {
                coototalcnt += blknnz[blkj + 1] - blknnz[blkj];
                ptroffset1[blkj] = coooffset;
                for (int bnnzid = 0; bnnzid < blknnz[blkj + 1] - blknnz[blkj]; bnnzid++)
                {
                    int row = (coo_compressed_Idx[coooffset + bnnzid] & num_f) >> 4;
                    int col = coo_compressed_Idx[coooffset + bnnzid] & num_b;
                    y[blki * BLOCK_SIZE + row] += Blockcoo_Val[coooffset + bnnzid] * x[x_offset + col];
                    int grow = blki * BLOCK_SIZE + row;
                }
                coooffset += blknnz[blkj + 1] - blknnz[blkj];
                break;
            }
            case 2:
            {
                ptroffset1[blkj] = ellwoffset;
                for (int ri = 0; ri < rowlength; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    for (int j = 0; j < tilewidth[blkj]; j++)
                    {
                        int ellcol = (ellwoffset + j * rowlength + ri) % 2 == 0 ? (ell_compressedIdx[(ellwoffset + j * rowlength + ri) / 2] & num_f) >> 4 : ell_compressedIdx[(ellwoffset + j * rowlength + ri) / 2] & num_b;
                        if (Blockell_Val[ellwoffset + j * rowlength + ri] != 0)
                        {
                            sum += Blockell_Val[ellwoffset + j * rowlength + ri] * x[x_offset + ellcol];
                        }
                    }
                    y[blki * BLOCK_SIZE + ri] += sum;
                }

                ellwoffset += tilewidth[blkj] * rowlength;
                break;
            }
            case 3:
            {
                ptroffset1[blkj] = hybwoffset;
                ptroffset2[blkj] = hybidxoffset;

                for (int ri = 0; ri < rowlength; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    for (int j = 0; j < tilewidth[blkj]; j++)
                    {
                        if (Blockhyb_Val[hybwoffset + j * rowlength + ri] != 0)
                        {
                            int hybcol = (j * rowlength + ri) % 2 == 0 ? (hybIdx[hybidxoffset + (j * rowlength + ri) / 2] & num_f) >> 4 : hybIdx[hybidxoffset + (j * rowlength + ri) / 2] & num_b;
                            sum += Blockhyb_Val[hybwoffset + j * rowlength + ri] * x[x_offset + hybcol];
                        }
                    }
                    y[blki * BLOCK_SIZE + ri] += sum;
                }
                int offset = hybwoffset + rowlength * tilewidth[blkj];
                hybidxoffset += (rowlength * tilewidth[blkj]) % 2 == 0 ? rowlength * tilewidth[blkj] / 2 : (rowlength * tilewidth[blkj] / 2) + 1;
                for (int i = 0; i < blknnz[blkj + 1] - blknnz[blkj] - tilewidth[blkj] * rowlength; i++)
                {
                    int rowidx = (hybIdx[hybidxoffset + i] & num_f) >> 4;
                    int colidx = hybIdx[hybidxoffset + i] & num_b;
                    y[blki * BLOCK_SIZE + rowidx] += Blockhyb_Val[offset + i] * x[x_offset + colidx];
                }

                hybwoffset += tilewidth[blkj] * rowlength + blknnz[blkj + 1] - blknnz[blkj] - tilewidth[blkj] * rowlength;
                hybidxoffset += blknnz[blkj + 1] - blknnz[blkj] - tilewidth[blkj] * rowlength;
                break;
            }
            case 4:
            {
                ptroffset1[blkj] = denseoffset;

                for (int ri = 0; ri < rowlength; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    for (int rj = ri * collength; rj < (ri + 1) * collength; rj++)
                    {
                        int densecol = rj % collength;
                        y[blki * BLOCK_SIZE + ri] += x[x_offset + densecol] * Blockdense_Val[denseoffset + densecol * rowlength + ri]; // Blockdense_Val[denseoffset +rj];
                    }
                }
                denseoffset += rowlength * collength;
                break;
            }
            case 5:
            {
                ptroffset1[blkj] = dnsrowoffset;
                for (int ri = dnsrowptr[blkj]; ri < dnsrowptr[blkj + 1]; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    for (int rj = 0; rj < collength; rj++)
                    {
                        sum += x[x_offset + rj] * Blockdenserow_Val[dnsrowoffset + (ri - dnsrowptr[blkj]) * collength + rj];
                    }
                    y[blki * BLOCK_SIZE + denserowid[ri]] += sum;
                }
                dnsrowoffset += blknnz[blkj + 1] - blknnz[blkj];
                break;
            }
            case 6:
            {
                ptroffset1[blkj] = dnscoloffset;
                for (int ri = 0; ri < rowlength; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    for (int rj = dnscolptr[blkj]; rj < dnscolptr[blkj + 1]; rj++)
                    {
                        sum += Blockdensecol_Val[dnscoloffset + (rj - dnscolptr[blkj]) * rowlength + ri] * x[x_offset + densecolid[rj]];
                    }
                    y[blki * BLOCK_SIZE + ri] += sum;
                }
                dnscoloffset += blknnz[blkj + 1] - blknnz[blkj];
                break;
            }
            }
        }
    }

    int errcount = 0;
    for (int i = 0; i < rowA; i++)
    {
        if (y[i] != y_golden[i])
        {

            errcount++;
            // printf("CPU ERROR %f    %f,%d\n",y[i],y_golden[i],i);
        }
    }
    printf(" Run CPU TileSpMV, errcount = %i\n", errcount);
}