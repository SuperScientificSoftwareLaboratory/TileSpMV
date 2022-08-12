#include "common.h"
#include "encode.h"
#include "format.h"

void convert_step1(Tile_matrix *matrix,
                   int rowA,
                   int colA,
                   MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA,
                   int *csrColIdxA,
                   MAT_VAL_TYPE *csrValA)
{

    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    int *tile_ptr = matrix->tile_ptr;

    unsigned thread = omp_get_max_threads();
    char *flag_g = (char *)malloc(thread * tilen * sizeof(char));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *flag = flag_g + thread_id * tilen;
        memset(flag, 0, tilen * sizeof(char));
        int start = blki * BLOCK_SIZE;
        int end = blki == tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;
        for (int j = csrRowPtrA[start]; j < csrRowPtrA[end]; j++)
        {
            int jc = csrColIdxA[j] / BLOCK_SIZE;
            if (flag[jc] == 0)
            {
                flag[jc] = 1;
                tile_ptr[blki]++;
            }
        }
    }
    free(flag_g);
}

void convert_step2(Tile_matrix *matrix,
                   unsigned char *tile_csr_ptr,
                   int rowA,
                   int colA,
                   MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA,
                   int *csrColIdxA,
                   MAT_VAL_TYPE *csrValA)
{

    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    int *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    unsigned thread = omp_get_max_threads();
    char *col_temp_g = (char *)malloc((thread * tilen) * sizeof(char));
    int *nnz_temp_g = (int *)malloc((thread * tilen) * sizeof(int));
    unsigned char *ptr_per_tile_g = (unsigned char *)malloc((thread * tilen * BLOCK_SIZE) * sizeof(unsigned char));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *col_temp = col_temp_g + thread_id * tilen;
        memset(col_temp, 0, tilen * sizeof(char));
        int *nnz_temp = nnz_temp_g + thread_id * tilen;
        memset(nnz_temp, 0, tilen * sizeof(int));
        unsigned char *ptr_per_tile = ptr_per_tile_g + thread_id * tilen * BLOCK_SIZE;
        memset(ptr_per_tile, 0, tilen * BLOCK_SIZE * sizeof(unsigned char));
        int pre_tile = tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int start = blki * BLOCK_SIZE;
        int end = blki == tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;

        for (int ri = 0; ri < rowlen; ri++)
        {
            for (int j = csrRowPtrA[start + ri]; j < csrRowPtrA[start + ri + 1]; j++)
            {
                int jc = csrColIdxA[j] / BLOCK_SIZE;
                col_temp[jc] = 1;
                nnz_temp[jc]++;
                ptr_per_tile[jc * BLOCK_SIZE + ri]++;
            }
        }

        int count = 0;
        for (int blkj = 0; blkj < tilen; blkj++)
        {
            if (col_temp[blkj] == 1)
            {
                tile_columnidx[pre_tile + count] = blkj;
                tile_nnz[pre_tile + count] = nnz_temp[blkj];
                for (int ri = 0; ri < rowlen; ri++)
                {
                    tile_csr_ptr[(pre_tile + count) * BLOCK_SIZE + ri] = ptr_per_tile[blkj * BLOCK_SIZE + ri];
                }
                count++;
            }
        }
    }
    free(col_temp_g);
    free(nnz_temp_g);
    free(ptr_per_tile_g);
}

void convert_step3(Tile_matrix *matrix,
                   unsigned char *tile_csr_ptr,
                   int rowA,
                   int colA,
                   MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA,
                   int *csrColIdxA,
                   MAT_VAL_TYPE *csrValA)

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

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int bi = 0; bi < tilenum_per_row; bi++)
        {
            int tile_id = tile_ptr[blki] + bi;
            int collen = tile_columnidx[tile_id] == tilen - 1 ? colA - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int nnztmp = tile_nnz[tile_id + 1] - tile_nnz[tile_id]; // the number of nnz of tile_id
            int nnzthreshold = rowlen * collen * 0.75;
            if (nnztmp >= nnzthreshold)
            {

                Format[tile_id] = 4;
                blknnz[tile_id] = rowlen * collen;
                dns_offset[tile_id] = rowlen * collen;
                continue;
            }
            else if (nnztmp <= COO_NNZ_TH)
            {
                {
                    Format[tile_id] = 1;
                    blknnz[tile_id] = nnztmp;
                    coo_offset[tile_id] = nnztmp;
                    new_coocount[tile_id] = nnztmp;
                    continue;
                }
            }
            else if (nnztmp % collen == 0 || nnztmp % rowlen == 0)
            {
                int dnsrowflag = 0;
                int numdnsrow = 0;
                int dnscolflag = 0;
                int numdnscol = 0;
                for (int ri = 0; ri < rowlen; ri++)
                {
                    if (tile_csr_ptr[tile_id * BLOCK_SIZE + ri] % collen != 0)
                    {
                        dnsrowflag = 0;
                        break;
                    }
                    else
                    {
                        if (tile_csr_ptr[tile_id * BLOCK_SIZE + ri] == collen)
                        {
                            dnsrowflag = 1;
                            numdnsrow++;
                        }
                    }
                }
                if (dnsrowflag == 1)
                {
                    Format[tile_id] = 5;
                    dnsrowptr[tile_id] = numdnsrow;
                    blknnz[tile_id] = numdnsrow * collen;
                    dnsrow_offset[tile_id] = numdnsrow * collen;
                    continue;
                }
                else
                {
                    int start = blki * BLOCK_SIZE;
                    int end = blki == tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;
                    int jc = tile_columnidx[tile_id];
                    unsigned char *dnscol_colidx_temp = (unsigned char *)malloc(nnztmp * sizeof(unsigned char));
                    memset(dnscol_colidx_temp, -1, nnztmp * sizeof(unsigned char));
                    unsigned char *col_flag = (unsigned char *)malloc(collen * sizeof(unsigned char));
                    memset(col_flag, 0, collen * sizeof(unsigned char));
                    for (int blkj = csrRowPtrA[start]; blkj < csrRowPtrA[end]; blkj++)
                    {
                        int jc_temp = csrColIdxA[blkj] / BLOCK_SIZE;
                        if (jc_temp == jc)
                        {
                            int col_temp = csrColIdxA[blkj] - jc * BLOCK_SIZE;
                            col_flag[col_temp]++;
                        }
                    }
                    for (int j = 0; j < collen; j++)
                    {
                        if (col_flag[j] % rowlen != 0)
                        {
                            dnscolflag = 0;
                            break;
                        }
                        else
                        {
                            if (col_flag[j] == rowlen)
                            {
                                dnscolflag = 1;
                                numdnscol++;
                            }
                        }
                    }
                    if (dnscolflag == 1)
                    {
                        Format[tile_id] = 6;
                        dnscolptr[tile_id] = numdnscol;
                        blknnz[tile_id] = numdnscol * rowlen;
                        dnscol_offset[tile_id] = numdnscol * rowlen;
                        continue;
                    }
                }
            }
            if (Format[tile_id] != 5 && Format[tile_id] != 6)
            {
                int bwidth = 0;
                for (int blkj = 0; blkj < rowlen; blkj++)
                {
                    if (bwidth < tile_csr_ptr[tile_id * BLOCK_SIZE + blkj])
                        bwidth = tile_csr_ptr[tile_id * BLOCK_SIZE + blkj];
                }
                double row_length_mean = ((double)nnztmp) / rowlen;
                double variance = 0.0;
                double row_length_skewness = 0.0;

                for (int row = 0; row < rowlen; ++row)
                {
                    int length = tile_csr_ptr[tile_id * BLOCK_SIZE + row];
                    double delta = (double)(length - row_length_mean);
                    variance += (delta * delta);
                    row_length_skewness += (delta * delta * delta);
                }
                variance /= rowlen;
                double row_length_std_dev = sqrt(variance);
                row_length_skewness = (row_length_skewness / rowlen) / pow(row_length_std_dev, 3.0);
                double row_length_variation = row_length_std_dev / row_length_mean;

                double ell_csr_threshold = 0.2;
                double csr_hyb_threshold = 1.0;

                if (row_length_variation <= ell_csr_threshold) // if variation is less than 0.2, then ELL
                {
                    Format[tile_id] = 2;
                    tilewidth[tile_id] = bwidth;
                    blknnz[tile_id] = bwidth * rowlen;
                    ell_offset[tile_id] = bwidth * rowlen;
                }
                else
                {
                    int hybwidth = bwidth;
                    int iopriorsize = (bwidth * rowlen) % 2 == 0 ? bwidth * rowlen * sizeof(MAT_VAL_TYPE) + (bwidth * rowlen * sizeof(char)) / 2 : bwidth * rowlen * sizeof(MAT_VAL_TYPE) + (bwidth * rowlen * sizeof(char)) / 2 + 1;
                    int ionextsize;
                    int coonextnum = 0;
                    int coopriornum = 0;
                    for (int wi = bwidth - 1; wi > 0; wi--)
                    {
                        coonextnum = 0;
                        for (int blkj = 0; blkj < rowlen; blkj++)
                        {
                            if (tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] > wi)
                            {
                                coonextnum += tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] - wi;
                            }
                        }
                        ionextsize = (wi * rowlen) % 2 == 0 ? wi * rowlen * sizeof(MAT_VAL_TYPE) + wi * rowlen * sizeof(char) / 2 + coonextnum * (sizeof(MAT_VAL_TYPE) + sizeof(char)) : wi * rowlen * sizeof(MAT_VAL_TYPE) + wi * rowlen * sizeof(char) / 2 + 1 + coonextnum * (sizeof(MAT_VAL_TYPE) + sizeof(char));
                        if (iopriorsize <= ionextsize)
                        {
                            hybwidth = wi + 1;
                            break;
                        }
                        else
                        {
                            hybwidth = wi;
                            iopriorsize = ionextsize;
                            coopriornum = coonextnum;
                        }
                    }

                    // if (row_length_variation >= csr_hyb_threshold && coopriornum <= 4) //&& coopriornum <= 4)  // if variation > 1.0, and the number of coo data <=4, then HYB
                    // {
                    //     Format[tile_id] = 3;
                    //     hyb_coocount[tile_id] = coopriornum;
                    //     tilewidth[tile_id] = hybwidth;
                    //     blknnz[tile_id] = coopriornum + hybwidth * rowlen;
                    //     hyb_offset[tile_id] = coopriornum + hybwidth * rowlen;
                    //     new_coocount[tile_id] = coopriornum;
                    // }
                    // else // else CSR
                    {
                        Format[tile_id] = 0;
                        blknnz[tile_id] = nnztmp;
                        csr_offset[tile_id] = nnztmp;
                        csrptr_offset[tile_id] = rowlen;
                    }
                }
            }
        }
    }
}

void convert_step4(Tile_matrix *matrix,
                   unsigned char *tile_csr_ptr,
                   unsigned char *Blockcsr_Col,
                   unsigned char *coo_colIdx,
                   unsigned char *coo_rowIdx,
                   unsigned char *ell_colIdx,
                   unsigned char *hyb_ellcolIdx,
                   unsigned char *hyb_coorowIdx,
                   MAT_VAL_TYPE *new_coo_value,
                   int *new_coo_rowidx,
                   int *new_coo_colidx,
                   int nnz_temp,
                   int tile_count_temp,
                   int rowA,
                   int colA,
                   MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA,
                   int *csrColIdxA,
                   MAT_VAL_TYPE *csrValA)

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
    unsigned char *Blockcsr_Ptr = matrix->Blockcsr_Ptr;
    MAT_VAL_TYPE *Blockcoo_Val = matrix->Blockcoo_Val;
    MAT_VAL_TYPE *Blockell_Val = matrix->Blockell_Val;
    MAT_VAL_TYPE *Blockhyb_Val = matrix->Blockhyb_Val;
    MAT_VAL_TYPE *Blockdense_Val = matrix->Blockdense_Val;
    MAT_VAL_TYPE *Blockdenserow_Val = matrix->Blockdenserow_Val;
    char *denserowid = matrix->denserowid;
    MAT_VAL_TYPE *Blockdensecol_Val = matrix->Blockdensecol_Val;
    char *densecolid = matrix->densecolid;

    unsigned thread = omp_get_max_threads();
    unsigned char *csr_colidx_temp_g = (unsigned char *)malloc((thread * nnz_temp) * sizeof(unsigned char));
    MAT_VAL_TYPE *csr_val_temp_g = (MAT_VAL_TYPE *)malloc((thread * nnz_temp) * sizeof(MAT_VAL_TYPE));
    int *tile_count_g = (int *)malloc(thread * tile_count_temp * sizeof(int));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {

        int thread_id = omp_get_thread_num();
        unsigned char *csr_colidx_temp = csr_colidx_temp_g + thread_id * nnz_temp;
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + thread_id * nnz_temp;
        int *tile_count = tile_count_g + thread_id * tile_count_temp;
        memset(csr_colidx_temp, 0, (nnz_temp) * sizeof(unsigned char));
        memset(csr_val_temp, 0, (nnz_temp) * sizeof(MAT_VAL_TYPE));
        memset(tile_count, 0, (tile_count_temp) * sizeof(int));
        int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int start = blki * BLOCK_SIZE;
        int end = blki == tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;
        for (int blkj = csrRowPtrA[start]; blkj < csrRowPtrA[end]; blkj++)
        {
            int jc_temp = csrColIdxA[blkj] / BLOCK_SIZE;
            for (int bi = 0; bi < tilenum_per_row; bi++)
            {
                int tile_id = tile_ptr[blki] + bi;
                int jc = tile_columnidx[tile_id];
                int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
                if (jc == jc_temp)
                {
                    csr_val_temp[pre_nnz + tile_count[bi]] = csrValA[blkj];
                    csr_colidx_temp[pre_nnz + tile_count[bi]] = csrColIdxA[blkj] - jc * BLOCK_SIZE;
                    tile_count[bi]++;
                    break;
                }
            }
        }
        for (int bi = 0; bi < tilenum_per_row; bi++)
        {
            int tile_id = tile_ptr[blki] + bi;
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
            int nnztmp = tile_nnz[tile_id + 1] - tile_nnz[tile_id]; // blknnz[tile_id+1] - blknnz[tile_id] ;
            int collen = tile_columnidx[tile_id] == tilen - 1 ? colA - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int format = Format[tile_id];
            switch (format)
            {
            case 0:
            {
                int offset = csr_offset[tile_id];
                int ptr_offset = csrptr_offset[tile_id];

                unsigned char *ptr_temp = tile_csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan_char(ptr_temp, rowlen);

                for (int ri = 0; ri < rowlen; ri++)
                {
                    int start = ptr_temp[ri];
                    int stop = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];
                    ;
                    for (int k = start; k < stop; k++)
                    {
                        unsigned char colidx = csr_colidx_temp[pre_nnz + k];
                        Blockcsr_Val[offset + k] = csr_val_temp[pre_nnz + k];
                        Blockcsr_Col[offset + k] = csr_colidx_temp[pre_nnz + k];
                    }
                    Blockcsr_Ptr[ptr_offset + ri] = ptr_temp[ri];
                }
                break;
            }
            case 1:
            {
                int colidx_temp = tile_columnidx[tile_id];

                int offset_new = new_coocount[tile_id];
                unsigned char *ptr_temp = tile_csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                for (int ri = 0; ri < rowlen; ri++)
                {
                    int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];
                    for (int j = ptr_temp[ri]; j < nnz_end; j++)
                    {
                        new_coo_rowidx[offset_new + j] = ri + blki * BLOCK_SIZE;
                        new_coo_value[offset_new + j] = csr_val_temp[pre_nnz + j];
                        new_coo_colidx[offset_new + j] = csr_colidx_temp[pre_nnz + j] + colidx_temp * BLOCK_SIZE;
                    }
                }
                int offset = coo_offset[tile_id];
                for (int ri = 0; ri < rowlen; ri++)
                {
                    int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];
                    for (int j = ptr_temp[ri]; j < nnz_end; j++)
                    {
                        unsigned char colidx = csr_colidx_temp[pre_nnz + j];
                        coo_rowIdx[offset + j] = ri;
                        Blockcoo_Val[offset + j] = csr_val_temp[pre_nnz + j];
                        coo_colIdx[offset + j] = csr_colidx_temp[pre_nnz + j];
                    }
                }

                break;
            }
            case 2:
            {
                int offset = ell_offset[tile_id];
                unsigned char *ptr_temp = tile_csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                for (int ri = 0; ri < rowlen; ri++)
                {
                    int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];

                    for (int j = ptr_temp[ri]; j < nnz_end; j++)
                    {
                        int colidx = csr_colidx_temp[pre_nnz + j];
                        int temp = j - ptr_temp[ri];
                        ell_colIdx[offset + temp * rowlen + ri] = csr_colidx_temp[pre_nnz + j];
                        Blockell_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz + j];
                    }
                }
                break;
            }
            case 3:
            {
                int colidx_temp = tile_columnidx[tile_id];
                int offset = hyb_offset[tile_id];
                unsigned char *ptr_temp = tile_csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                int offset_new = new_coocount[tile_id];

                int coocount_case1 = 0;
                int coocount_case2 = 0;
                for (int ri = 0; ri < rowlen; ri++)
                {
                    int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];
                    int stop = (nnz_end - ptr_temp[ri]) <= tilewidth[tile_id] ? nnz_end : ptr_temp[ri] + tilewidth[tile_id];

                    for (int j = ptr_temp[ri]; j < stop; j++)
                    {
                        int colidx = csr_colidx_temp[pre_nnz + j];

                        int temp = j - ptr_temp[ri];
                        hyb_ellcolIdx[offset + temp * rowlen + ri] = csr_colidx_temp[pre_nnz + j];
                        Blockhyb_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz + j];
                    }

                    for (int k = stop; k < nnz_end; k++)
                    {
                        unsigned char colidx = csr_colidx_temp[pre_nnz + k];
                        Blockhyb_Val[offset + tilewidth[tile_id] * rowlen + coocount_case1] = csr_val_temp[pre_nnz + k];
                        hyb_ellcolIdx[offset + tilewidth[tile_id] * rowlen + coocount_case1] = csr_colidx_temp[pre_nnz + k];
                        hyb_coorowIdx[hyb_coocount[tile_id] + coocount_case1] = ri;
                        coocount_case1++;
                    }

                    for (int k = stop; k < nnz_end; k++)
                    {
                        new_coo_value[offset_new + coocount_case2] = csr_val_temp[pre_nnz + k];
                        new_coo_colidx[offset_new + coocount_case2] = csr_colidx_temp[pre_nnz + k] + colidx_temp * BLOCK_SIZE;
                        new_coo_rowidx[offset_new + coocount_case2] = ri + blki * BLOCK_SIZE;
                        coocount_case2++;
                    }
                }
                break;
            }
            case 4:
            {
                int offset = dns_offset[tile_id];
                unsigned char *ptr_temp = tile_csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                for (int ri = 0; ri < rowlen; ri++)
                {
                    int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];

                    for (int j = ptr_temp[ri]; j < nnz_end; j++)
                    {
                        unsigned char colidx = csr_colidx_temp[pre_nnz + j];
                        Blockdense_Val[offset + csr_colidx_temp[pre_nnz + j] * rowlen + ri] = csr_val_temp[pre_nnz + j];
                    }
                }

                break;
            }
            case 5:
            {
                int offset = dnsrow_offset[tile_id];
                int rowoffset = dnsrowptr[tile_id];
                unsigned char *ptr_temp = tile_csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                int dnsriid = 0;
                for (int ri = 0; ri < rowlen; ri++)
                {
                    int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];
                    if (nnz_end - ptr_temp[ri] == collen)
                    {
                        denserowid[rowoffset + dnsriid] = ri;
                        dnsriid++;
                        for (int j = ptr_temp[ri]; j < nnz_end; j++)
                        {
                            unsigned char colidx = csr_colidx_temp[pre_nnz + j];
                            Blockdenserow_Val[offset + j] = csr_val_temp[pre_nnz + j];
                        }
                    }
                }
                break;
            }
            case 6:
            {
                int offset = dnscol_offset[tile_id];
                int coloffset = dnscolptr[tile_id];
                unsigned char *ptr_temp = tile_csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                int dnsciid = 0;
                for (int j = ptr_temp[0]; j < ptr_temp[1]; j++)
                {
                    int ci = csr_colidx_temp[pre_nnz + j];
                    densecolid[coloffset + dnsciid] = ci;
                    dnsciid++;
                }
                for (int ri = 0; ri < rowlen; ri++)
                {
                    int nnz_end = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];

                    for (int j = ptr_temp[ri]; j < nnz_end; j++)
                    {
                        int temp = j - ptr_temp[ri];
                        unsigned char colidx = csr_colidx_temp[pre_nnz + j]; // temp;
                        Blockdensecol_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz + j];
                    }
                }
                break;
            }
            default:
                break;
            }
        }
    }
    free(csr_colidx_temp_g);
    free(csr_val_temp_g);
    free(tile_count_g);
}

void Tile_create(Tile_matrix *matrix,
                 int rowA,
                 int colA,
                 MAT_PTR_TYPE nnzA,
                 MAT_PTR_TYPE *csrRowPtrA,
                 int *csrColIdxA,
                 MAT_VAL_TYPE *csrValA)
{

    struct timeval t1, t2;
    double time_conversion = 0;

    matrix->tilem = rowA % BLOCK_SIZE == 0 ? rowA / BLOCK_SIZE : (rowA / BLOCK_SIZE) + 1;
    matrix->tilen = colA % BLOCK_SIZE == 0 ? colA / BLOCK_SIZE : (colA / BLOCK_SIZE) + 1;
    matrix->tile_ptr = (int *)malloc((matrix->tilem + 1) * sizeof(int));
    memset(matrix->tile_ptr, 0, (matrix->tilem + 1) * sizeof(int));

#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif
    convert_step1(matrix,
                  rowA, colA, nnzA,
                  csrRowPtrA, csrColIdxA, csrValA);

#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

    exclusive_scan(matrix->tile_ptr, matrix->tilem + 1);
    matrix->tilenum = matrix->tile_ptr[matrix->tilem];
    int tilenum = matrix->tilenum;
    printf("\n  The number of tile = %i\n", tilenum);

    matrix->tile_columnidx = (int *)malloc(tilenum * sizeof(int));
    memset(matrix->tile_columnidx, 0, tilenum * sizeof(int));

    matrix->tile_nnz = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->tile_nnz, 0, (tilenum + 1) * sizeof(int));
    unsigned char *tile_csr_ptr = (unsigned char *)malloc((tilenum * BLOCK_SIZE) * sizeof(unsigned char));
    memset(tile_csr_ptr, 0, (tilenum * BLOCK_SIZE) * sizeof(unsigned char));

#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif
    convert_step2(matrix, tile_csr_ptr,
                  rowA, colA, nnzA,
                  csrRowPtrA, csrColIdxA, csrValA);
#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif
    exclusive_scan(matrix->tile_nnz, tilenum + 1);

    matrix->Format = (char *)malloc(tilenum * sizeof(char));
    memset(matrix->Format, 0, tilenum * sizeof(char));
    matrix->blknnz = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->blknnz, 0, (tilenum + 1) * sizeof(int));
    matrix->blknnznnz = (unsigned char *)malloc((tilenum + 1) * sizeof(unsigned char));
    memset(matrix->blknnznnz, 0, (tilenum + 1) * sizeof(unsigned char));

    matrix->dnsrowptr = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->dnsrowptr, 0, (tilenum + 1) * sizeof(int));
    matrix->dnscolptr = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->dnscolptr, 0, (tilenum + 1) * sizeof(int));
    matrix->tilewidth = (char *)malloc(tilenum * sizeof(char));
    memset(matrix->tilewidth, 0, tilenum * sizeof(char));
    matrix->csr_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->csr_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->csrptr_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->csrptr_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->coo_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->coo_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->ell_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->ell_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->hyb_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->hyb_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->hyb_coocount = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->hyb_coocount, 0, (tilenum + 1) * sizeof(int));
    matrix->dns_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->dns_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->dnsrow_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->dnsrow_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->dnscol_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->dnscol_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->new_coocount = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->new_coocount, 0, (tilenum + 1) * sizeof(int));

#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif

    convert_step3(matrix, tile_csr_ptr,
                  rowA, colA, nnzA,
                  csrRowPtrA, csrColIdxA, csrValA);
#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

    exclusive_scan(matrix->csr_offset, tilenum + 1);
    exclusive_scan(matrix->csrptr_offset, tilenum + 1);
    exclusive_scan(matrix->coo_offset, tilenum + 1);
    exclusive_scan(matrix->ell_offset, tilenum + 1);
    exclusive_scan(matrix->hyb_offset, tilenum + 1);
    exclusive_scan(matrix->dns_offset, tilenum + 1);
    exclusive_scan(matrix->dnsrow_offset, tilenum + 1);
    exclusive_scan(matrix->dnscol_offset, tilenum + 1);
    exclusive_scan(matrix->dnsrowptr, tilenum + 1);
    exclusive_scan(matrix->dnscolptr, tilenum + 1);
    exclusive_scan(matrix->hyb_coocount, tilenum + 1);
    exclusive_scan(matrix->new_coocount, tilenum + 1);

    matrix->csrsize = 0;
    matrix->coosize = 0;
    matrix->ellsize = 0;
    matrix->hybsize = 0;
    matrix->hybellsize = 0;
    matrix->hybcoosize = 0;
    matrix->dnssize = 0;
    matrix->dnsrowsize = 0;
    matrix->dnscolsize = 0;

    matrix->hybcoosize = matrix->hyb_coocount[tilenum];
    matrix->csrptrlen = 0;

    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        int rowlength = blki == matrix->tilem - 1 ? rowA - (matrix->tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int rowbnum = matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki];
        for (int bi = 0; bi < rowbnum; bi++)
        {
            int tile_id = matrix->tile_ptr[blki] + bi;
            char format = matrix->Format[tile_id];
            switch (format)
            {
            case 0: // csr
                matrix->csrsize += matrix->blknnz[tile_id];
                matrix->csrptrlen += rowlength;
                break;

            case 1: // coo
                matrix->coosize += matrix->blknnz[tile_id];
                break;
            case 2: // ell
                matrix->ellsize += matrix->blknnz[tile_id];
                break;
            case 3: // hyb
                matrix->hybsize += matrix->blknnz[tile_id];
                matrix->hybellsize += matrix->tilewidth[tile_id] * rowlength;
                break;
            case 4:
                matrix->dnssize += matrix->blknnz[tile_id];
                break;
            case 5:
                matrix->dnsrowsize += matrix->blknnz[tile_id];
                break;
            case 6:
                matrix->dnscolsize += matrix->blknnz[tile_id];
                break;

            default:
                break;
            }
        }
    }

    for (int i = 0; i < tilenum + 1; i++)
        matrix->blknnznnz[i] = matrix->blknnz[i];

    exclusive_scan(matrix->blknnz, tilenum + 1);

    // CSR
    matrix->Blockcsr_Val = (MAT_VAL_TYPE *)malloc((matrix->csrsize) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Blockcsr_Val, 0, (matrix->csrsize) * sizeof(MAT_VAL_TYPE));
    unsigned char *Blockcsr_Col_tmp = (unsigned char *)malloc((matrix->csrsize) * sizeof(unsigned char));
    memset(Blockcsr_Col_tmp, 0, (matrix->csrsize) * sizeof(unsigned char));
    matrix->Blockcsr_Ptr = (unsigned char *)malloc((matrix->csrptrlen) * sizeof(unsigned char));
    memset(matrix->Blockcsr_Ptr, 0, (matrix->csrptrlen) * sizeof(unsigned char));
    int compressed_csr_size = matrix->csrsize % 2 == 0 ? matrix->csrsize / 2 : matrix->csrsize / 2 + 1;
    matrix->csr_compressedIdx = (unsigned char *)malloc((compressed_csr_size) * sizeof(unsigned char));
    memset(matrix->csr_compressedIdx, 0, (compressed_csr_size) * sizeof(unsigned char));

    // COO
    matrix->Blockcoo_Val = (MAT_VAL_TYPE *)malloc((matrix->coosize) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Blockcoo_Val, 0, (matrix->coosize) * sizeof(MAT_VAL_TYPE));
    unsigned char *coo_colIdx_tmp = (unsigned char *)malloc((matrix->coosize) * sizeof(unsigned char));
    memset(coo_colIdx_tmp, 0, (matrix->coosize) * sizeof(unsigned char));
    unsigned char *coo_rowIdx_tmp = (unsigned char *)malloc((matrix->coosize) * sizeof(unsigned char));
    memset(coo_rowIdx_tmp, 0, (matrix->coosize) * sizeof(unsigned char));

    matrix->coo_compressed_Idx = (unsigned char *)malloc((matrix->coosize) * sizeof(unsigned char));
    memset(matrix->coo_compressed_Idx, 0, (matrix->coosize) * sizeof(unsigned char));

    // ELL
    matrix->Blockell_Val = (MAT_VAL_TYPE *)malloc((matrix->ellsize) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Blockell_Val, 0, (matrix->ellsize) * sizeof(MAT_VAL_TYPE));
    unsigned char *ell_colIdx_tmp = (unsigned char *)malloc((matrix->ellsize) * sizeof(unsigned char));
    memset(ell_colIdx_tmp, 0, sizeof(unsigned char) * (matrix->ellsize));
    int ell_compressed_size = matrix->ellsize % 2 == 0 ? matrix->ellsize / 2 : matrix->ellsize / 2 + 1;
    matrix->ell_compressedIdx = (unsigned char *)malloc(ell_compressed_size * sizeof(unsigned char));
    memset(matrix->ell_compressedIdx, 0, sizeof(unsigned char) * ell_compressed_size);

    // HYB
    matrix->Blockhyb_Val = (MAT_VAL_TYPE *)malloc((matrix->hybellsize + matrix->hybcoosize) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Blockhyb_Val, 0, (matrix->hybellsize + matrix->hybcoosize) * sizeof(MAT_VAL_TYPE));
    unsigned char *hyb_ellcolIdx_tmp = (unsigned char *)malloc((matrix->hybellsize + matrix->hybcoosize) * sizeof(unsigned char));
    memset(hyb_ellcolIdx_tmp, 0, sizeof(unsigned char) * (matrix->hybellsize + matrix->hybcoosize));
    unsigned char *hyb_coorowIdx_tmp = (unsigned char *)malloc((matrix->hybcoosize) * sizeof(unsigned char));
    memset(hyb_coorowIdx_tmp, 0, sizeof(unsigned char) * (matrix->hybcoosize));

    int hyb_compressed_size = matrix->hybellsize % 2 == 0 ? matrix->hybellsize / 2 : (matrix->hybellsize / 2) + 1;
    matrix->hybIdx = (unsigned char *)malloc((hyb_compressed_size + matrix->hybcoosize) * sizeof(unsigned char));
    memset(matrix->hybIdx, 0, (hyb_compressed_size + matrix->hybcoosize) * sizeof(unsigned char));

    // dns
    matrix->Blockdense_Val = (MAT_VAL_TYPE *)malloc((matrix->dnssize) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Blockdense_Val, 0, matrix->dnssize * sizeof(MAT_VAL_TYPE));

    // dnsrow
    matrix->Blockdenserow_Val = (MAT_VAL_TYPE *)malloc((matrix->dnsrowsize) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Blockdenserow_Val, 0, (matrix->dnsrowsize) * sizeof(MAT_VAL_TYPE));
    matrix->denserowid = (char *)malloc(matrix->dnsrowptr[tilenum] * sizeof(char));
    memset(matrix->denserowid, 0, matrix->dnsrowptr[tilenum] * sizeof(char));

    // dnscol
    matrix->Blockdensecol_Val = (MAT_VAL_TYPE *)malloc((matrix->dnscolsize) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Blockdensecol_Val, 0, (matrix->dnscolsize) * sizeof(MAT_VAL_TYPE));
    matrix->densecolid = (char *)malloc(matrix->dnscolptr[tilenum] * sizeof(char));
    memset(matrix->densecolid, 0, matrix->dnscolptr[tilenum] * sizeof(char));

    // deferred coo
    matrix->coototal = matrix->new_coocount[tilenum];

    MAT_VAL_TYPE *new_coo_value_temp = (MAT_VAL_TYPE *)malloc(matrix->coototal * sizeof(MAT_VAL_TYPE));
    memset(new_coo_value_temp, 0, (matrix->coototal) * sizeof(MAT_VAL_TYPE));
    int *new_coo_rowidx_temp = (int *)malloc((matrix->coototal) * sizeof(int));
    memset(new_coo_rowidx_temp, 0, (matrix->hybcoosize + matrix->coosize) * sizeof(int));

    int *new_coo_colidx_temp = (int *)malloc((matrix->coototal) * sizeof(int));
    memset(new_coo_colidx_temp, 0, (matrix->coototal) * sizeof(int));

    int nnz_temp = 0;
    int tile_count_temp = 0;
    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        int start = blki * BLOCK_SIZE;
        int end = blki == matrix->tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;
        nnz_temp = nnz_temp < csrRowPtrA[end] - csrRowPtrA[start] ? csrRowPtrA[end] - csrRowPtrA[start] : nnz_temp;
        tile_count_temp = tile_count_temp < matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki] ? matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki] : tile_count_temp;
    }
#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif

    convert_step4(matrix, tile_csr_ptr,
                  Blockcsr_Col_tmp,
                  coo_colIdx_tmp, coo_rowIdx_tmp,
                  ell_colIdx_tmp,
                  hyb_ellcolIdx_tmp, hyb_coorowIdx_tmp,
                  new_coo_value_temp, new_coo_rowidx_temp, new_coo_colidx_temp,
                  nnz_temp, tile_count_temp,
                  rowA, colA, nnzA,
                  csrRowPtrA, csrColIdxA, csrValA);

#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

    // deferred coo
    matrix->deferredcoo_val = (MAT_VAL_TYPE *)malloc(matrix->coototal * sizeof(MAT_VAL_TYPE));
    memset(matrix->deferredcoo_val, 0, (matrix->coototal) * sizeof(MAT_VAL_TYPE));

    matrix->deferredcoo_colidx = (int *)malloc((matrix->coototal) * sizeof(int));
    memset(matrix->deferredcoo_colidx, 0, (matrix->coototal) * sizeof(int));

    matrix->deferredcoo_ptr = (MAT_PTR_TYPE *)malloc((rowA + 1) * sizeof(MAT_PTR_TYPE));
    memset(matrix->deferredcoo_ptr, 0, (rowA + 1) * sizeof(MAT_PTR_TYPE));

    //  convert COO to CSR
    int nthreads = omp_get_max_threads();

    int *count_temp_g = (int *)malloc((rowA * nthreads) * sizeof(int));
    memset(count_temp_g, 0, (rowA * nthreads) * sizeof(int));
    int nnz_ave = matrix->coototal / nthreads;

#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif

#pragma omp parallel for
    for (int i = 0; i < nthreads; i++)
    {
        int end = i == nthreads - 1 ? matrix->coototal : (i + 1) * nnz_ave;
        for (int j = i * nnz_ave; j < end; j++)
        {
            int rowidx = new_coo_rowidx_temp[j];
            ++count_temp_g[i * rowA + rowidx];
        }
    }

    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < nthreads; j++)
        {
            matrix->deferredcoo_ptr[i] += count_temp_g[j * rowA + i];
        }
    }
    exclusive_scan(matrix->deferredcoo_ptr, rowA + 1);

    int *row_cnt = (int *)malloc(rowA * sizeof(int));
    memset(row_cnt, 0, rowA * sizeof(int));

    for (int i = 0; i < matrix->coototal; ++i)
    {
        int rowidx = new_coo_rowidx_temp[i];
        int nnz_offset = matrix->deferredcoo_ptr[rowidx];
        matrix->deferredcoo_val[nnz_offset + row_cnt[rowidx]] = new_coo_value_temp[i];
        matrix->deferredcoo_colidx[nnz_offset + row_cnt[rowidx]] = new_coo_colidx_temp[i];
        row_cnt[rowidx]++;
    }

#pragma omp parallel for

    for (int i = 0; i < rowA; i++)
    {
        int nnz_offset = matrix->deferredcoo_ptr[i];
        int length = matrix->deferredcoo_ptr[i + 1] - matrix->deferredcoo_ptr[i];

        quick_sort_key_val_pair(matrix->deferredcoo_colidx + nnz_offset, matrix->deferredcoo_val + nnz_offset, length);
    }

#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

#if FORMAT_CONVERSION
    printf("CSR -> Tile time = %4.2f ms\n", time_conversion);
#endif

    // cpmpress the idx

    encode(Blockcsr_Col_tmp, matrix->csr_compressedIdx, matrix->csrsize, 0);

    int count = 0;
    for (int i = 0; i < matrix->coosize; i++)
    {
        matrix->coo_compressed_Idx[count] = (coo_rowIdx_tmp[i] << 4) + coo_colIdx_tmp[i];
        count++;
    }

    encode(ell_colIdx_tmp, matrix->ell_compressedIdx, matrix->ellsize, 0);

    int blkoffset = 0;
    int hybcoonnzsum = 0;
    int hc_offset = 0;
    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        int rowlength = blki == matrix->tilem - 1 ? rowA - (matrix->tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int blkj = matrix->tile_ptr[blki]; blkj < matrix->tile_ptr[blki + 1]; blkj++)
        {
            if (matrix->Format[blkj] == 3)
            {
                encode(hyb_ellcolIdx_tmp + blkoffset, matrix->hybIdx, matrix->tilewidth[blkj] * rowlength, hc_offset);

                hc_offset += (rowlength * matrix->tilewidth[blkj]) % 2 == 0 ? (rowlength * matrix->tilewidth[blkj]) / 2 : (rowlength * matrix->tilewidth[blkj] / 2) + 1;
                for (int i = 0; i < matrix->blknnz[blkj + 1] - matrix->blknnz[blkj] - matrix->tilewidth[blkj] * rowlength; i++)
                {
                    matrix->hybIdx[hc_offset + i] = (hyb_coorowIdx_tmp[hybcoonnzsum + i] << 4) + hyb_ellcolIdx_tmp[blkoffset + rowlength * matrix->tilewidth[blkj] + i];
                }
                hybcoonnzsum += matrix->blknnz[blkj + 1] - matrix->blknnz[blkj] - matrix->tilewidth[blkj] * rowlength;

                blkoffset += matrix->tilewidth[blkj] * rowlength + matrix->blknnz[blkj + 1] - matrix->blknnz[blkj] - matrix->tilewidth[blkj] * rowlength;

                hc_offset += matrix->blknnz[blkj + 1] - matrix->blknnz[blkj] - matrix->tilewidth[blkj] * rowlength;
            }
        }
    }

    free(new_coo_rowidx_temp);
    free(new_coo_colidx_temp);
    free(new_coo_value_temp);
    free(Blockcsr_Col_tmp);
    free(coo_colIdx_tmp);
    free(coo_rowIdx_tmp);
    free(ell_colIdx_tmp);
    free(hyb_ellcolIdx_tmp);
    free(hyb_coorowIdx_tmp);
    free(tile_csr_ptr);
}
