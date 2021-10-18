#ifndef FORMAT_TRANS
#define FORMAT_TRANS
#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"


//calculate the number of non-empty tiles of matrix A
void step1_kernel(Beidou_Tile_Matrix *matrix)

// (int m, int n, MAT_PTR_TYPE *rowpointer, int *columnidx, 
//                   int tilem, int tilen, MAT_PTR_TYPE *tile_ptr, int *numtile)
{
    int *rowpointer=matrix->rowpointer;
    int m = matrix->m;
    int n = matrix->n;
    int *columnidx = matrix->columnidx;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int numtile = matrix->numtile;
    unsigned thread = omp_get_max_threads();
    printf("threads=%i\n",thread);
    char *flag_g=(char *)malloc(thread*tilen * sizeof(char));

    #pragma omp parallel for  
    for (int blki = 0; blki < tilem; blki ++)
	{
        int thread_id = omp_get_thread_num();
     //   printf("id =%d\n",thread_id);
        char *flag = flag_g+ thread_id * tilen;
        memset(flag,0,tilen * sizeof(char));
        int start = blki *BLOCK_SIZE;
        int end = blki == tilem-1 ?  m : (blki+1)* BLOCK_SIZE ;
		for (int j = rowpointer[start]; j < rowpointer[end]; j ++)
        {
            int jc = columnidx[j] / BLOCK_SIZE;
            if (flag[jc]==0)
            {
                flag[jc]=1;
                tile_ptr[blki]++;
            }
	    } 
     //   free(flag);
	}
    free(flag_g);
}

void step2_kernel(Beidou_Tile_Matrix *matrix)


// (int rowA, int coA, int *rowpointerA, int *columnindexA, 
//                   int tilem, int tilenA, MAT_PTR_TYPE *tile_ptr, int *tile_columnidx, int numtileA)
{

    int *rowpointer=matrix->rowpointer;
    int *columnidx = matrix->columnidx;
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int numtile = matrix->numtile;
    int *tile_columnidx = matrix->columnidx;


    int colid =0;
    char *flag=(char *)malloc(tilen * sizeof(char));
    for (int i=0;i<tilem;i++)
	{
        memset(flag,0,tilen*sizeof(char));
        int start= i*BLOCK_SIZE;
        int end = i== tilem-1 ?  m : (i+1)*BLOCK_SIZE ;
		for (int j=rowpointer[start];j< rowpointer[end];j++)
        {
            int jc=columnidx[j]/BLOCK_SIZE;
            if (flag[jc]==0)
            {
                flag[jc]=1;
                tile_ptr[i+1]++;
                tile_columnidx[colid]=jc;
                colid++;
            }
	    } 
	}
    for (int i=1;i<tilem+1;i++)
	{
		tile_ptr[i] += tile_ptr[i-1];
	}
}




//determine the tile structure (tileptr , tile columnidx and tile_nnz) of matrix A. 

void step2_kernel_new
(Beidou_Tile_Matrix *matrix, unsigned char *tile_csr_ptr)

// (int m, int n, int *rowpointer, int *columnidx, 
//                   int tilem, int tilen, MAT_PTR_TYPE *tile_ptr, int *tile_columnidx, int *tile_nnz, 
//                   unsigned char *tile_csr_ptr, int numtile)
{
    int m = matrix->m;
    int n = matrix->n;
    int *rowpointer=matrix->rowpointer;
    int *columnidx = matrix->columnidx;
  
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    int numtile = matrix->numtile;
    unsigned thread = omp_get_max_threads();
    
    char *col_temp_g=(char *)malloc((thread * tilen) * sizeof(char));

    int *nnz_temp_g=(int *)malloc((thread * tilen) * sizeof(int));

    unsigned char *ptr_per_tile_g = (unsigned char *)malloc((thread * tilen * BLOCK_SIZE) * sizeof(unsigned char));

    #pragma omp parallel for  
    for (int blki = 0; blki < tilem; blki ++)
	{
        int thread_id = omp_get_thread_num();
        char *col_temp = col_temp_g + thread_id * tilen;
        memset(col_temp,0,tilen * sizeof(char));
        int *nnz_temp = nnz_temp_g + thread_id * tilen;
        memset(nnz_temp,0,tilen * sizeof(int));
        unsigned char *ptr_per_tile = ptr_per_tile_g + thread_id * tilen * BLOCK_SIZE;
        memset(ptr_per_tile, 0, tilen * BLOCK_SIZE * sizeof(unsigned char));
        int  pre_tile = tile_ptr[blki];
        int rowlen = blki==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int start= blki * BLOCK_SIZE;
        int end = blki==tilem-1 ?  m : (blki +1)*BLOCK_SIZE ;

        for (int ri=0 ; ri < rowlen ; ri ++)
        {
            for (int j=rowpointer[start + ri];j<rowpointer[start + ri +1];j++)
            {
                int jc = columnidx[j]/BLOCK_SIZE;
                col_temp[jc] = 1;
                nnz_temp[jc] ++;
                ptr_per_tile[jc * BLOCK_SIZE + ri] ++;
            }

        }
        
        int count =0;
        for (int blkj=0 ;blkj < tilen; blkj++)
        {
            if (col_temp[blkj] == 1)
            {
                tile_columnidx[pre_tile + count] = blkj;
                tile_nnz[pre_tile + count] = nnz_temp[blkj];
                for (int ri =0; ri < rowlen ; ri ++)
                {
                    tile_csr_ptr[(pre_tile + count) * BLOCK_SIZE + ri] = ptr_per_tile[blkj * BLOCK_SIZE + ri];
                }
                count ++;
            }
        }
    }
    free(col_temp_g);
    free(nnz_temp_g);
    free(ptr_per_tile_g);
}


void step3_kernel_new(Beidou_Tile_Matrix *matrix, int *new_coocount)

{

    int *rowpointer=matrix->rowpointer;
    int *columnidx = matrix->columnidx;
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int numtile = matrix->numtile;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    int *blknnz = matrix->blknnz;
    char *blkwidth = matrix->blkwidth;
    int *denserowptr = matrix->denserowptr;
    int *densecolptr = matrix->densecolptr;

    unsigned char *tile_csr_ptr = matrix->csr_ptr;
    int *hyb_coocount = matrix->hyb_coocount;
    int *csr_offset = matrix->csr_offset; 
    int *csrptr_offset = matrix->csrptr_offset;
    int *coo_offset = matrix->coo_offset;
    int *ell_offset = matrix->ell_offset;
    int *hyb_offset = matrix->hyb_offset;
    int *dns_offset = matrix->dns_offset;
    int *dnsrow_offset = matrix->dnsrow_offset;
    int *dnscol_offset = matrix->dnscol_offset;


    #pragma omp parallel for  
    for (int blki=0;blki<tilem;blki++)
    {
        int tilenum_per_row=tile_ptr[blki+1]-tile_ptr[blki];
        int rowlen= blki==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
        for (int bi=0;bi<tilenum_per_row;bi++)
        {
            int collen = tile_columnidx[tile_ptr[blki]+bi] == tilen-1 ? n - (tilen-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            int tile_id = tile_ptr[blki]+bi;
            int tilennz = tile_nnz[tile_id +1] - tile_nnz[tile_id];
            int nnzthreshold = rowlen * collen * 0.5 ;
            // if (1)
            // {
            //             Format[tile_id] =0 ;
            //             blknnz[tile_id] = tilennz ;
            //             csr_offset[tile_id] = tilennz;
            //             csrptr_offset[tile_id] = rowlen;
            // }

            if (tilennz >= nnzthreshold)  //if the number of nnz is more than 128, then dense
            {
                Format[tile_id] = 4 ;
                blknnz[tile_id] = rowlen * collen;
                dns_offset[tile_id] = rowlen * collen;
                continue;
            }
            if (tilennz <= COO_THRESHOLD) //else if the number of nnz is less than 12, then coo
            {
                Format[tile_id] = 1 ;
                blknnz[tile_id] = tilennz;
                coo_offset[tile_id] = tilennz;
                new_coocount[tile_id] = tilennz;
                continue;
            }

            else if (tilennz % collen ==0 || tilennz % rowlen ==0)
            {
                int dnsrowflag =0 ;
                int numdnsrow =0;
                int dnscolflag =0;
                int numdnscol =0;
                for (int ri=0;ri < rowlen ;ri++)
                {
                    if (tile_csr_ptr[tile_id * BLOCK_SIZE + ri] % collen !=0)
                    {
                        dnsrowflag =0;
                        break;
                    }
                    else 
                    {
                        if (tile_csr_ptr[tile_id * BLOCK_SIZE + ri]  == collen)
                        {
                            dnsrowflag =1;
                            numdnsrow ++ ;
                        }
                    }
                    
                }
                if (dnsrowflag  == 1)
                {                    
                    Format[tile_id] = 5 ;   //Dense Row
                    denserowptr[tile_id] = numdnsrow ;
                    blknnz[tile_id] = numdnsrow * collen;
                    dnsrow_offset[tile_id] = numdnsrow * collen;
                    continue;
                }
                else 
                {
                    int start = blki*BLOCK_SIZE;
                    int end = blki==tilem-1 ?  m : (blki+1)*BLOCK_SIZE ;
                    int jc = tile_columnidx[tile_id];
                    unsigned char *dnscol_colidx_temp= (unsigned char *)malloc(tilennz * sizeof(unsigned char));
                    memset(dnscol_colidx_temp, -1, tilennz * sizeof(unsigned char));
                  //  int k=0;
                    unsigned char *col_flag =(unsigned char *)malloc(collen * sizeof(unsigned char));
                    memset(col_flag, 0, collen * sizeof(unsigned char));
                    for (int blkj = rowpointer[start]; blkj < rowpointer[end]; blkj ++)
                    {
                        int jc_temp = columnidx[blkj]/BLOCK_SIZE;
                        if (jc_temp == jc)
                        {
                            int col_temp = columnidx[blkj] - jc * BLOCK_SIZE;
                            col_flag[col_temp] ++;
//                            dnscol_colidx_temp[k]= columnindexA[blkj] - jc * BLOCK_SIZE;
                            // if (tile_id == 389)
                            //     printf("colidx = %i\n", dnscol_colidx_temp[k]);
                        //    k++;
                        }

                    }
                    for (int j =0; j < collen; j ++)
                    {
                        if (col_flag[j] % rowlen !=0)
                        {
                            dnscolflag =0;
                            break;
                        }
                        else 
                        {
                           if (col_flag[j] == rowlen)
                            {
                                dnscolflag =1;
                                numdnscol ++ ;
                            }
                        }
                    }
                    if (dnscolflag  == 1)
                    {                    
                //        printf("numdnscol = %i\n", numdnscol);
                        Format[tile_id] = 6 ;   //Dense Col
                        densecolptr[tile_id] = numdnscol ;
                        blknnz[tile_id] = numdnscol * rowlen;
                        dnscol_offset[tile_id] = numdnscol * rowlen;
                        continue;
                    }            


                //     unsigned char *trans_ptr= (unsigned char *)malloc(collen * sizeof(unsigned char));
                //     memset(trans_ptr, 0, collen * sizeof(unsigned char));
                //     for (int ni =0; ni < tilennz; ni ++)
                //     {
                //         int coltemp = dnscol_colidx_temp[ni];
                //         trans_ptr[coltemp]++;
                //     }
                //     for (int ri=0;ri < rowlen ;ri++)
                //     {
                //         if (trans_ptr[ri] % rowlen !=0)
                //         {
                //             dnscolflag =0;
                //             break;
                //         }
                //         else 
                //         {
                //            if (trans_ptr[ri] == rowlen)
                //             {
                //                 dnscolflag =1;
                //                 numdnscol ++ ;
                //             }
                //         }
                        
                //     }
                //     if (dnscolflag  == 1)
                //     {                    
                // //        printf("numdnscol = %i\n", numdnscol);
                //         Format[tile_id] = 6 ;   //Dense Col
                //         densecolptr[tile_id] = numdnscol ;
                //         blknnz[tile_id] = numdnscol * rowlen;
                //         dnscol_offset[tile_id] = numdnscol * rowlen;
                //         continue;
                //     }                    
                }
            }
            if (Format[tile_id] != 5 && Format[tile_id] !=6)
            {
                int bwidth=0;
                for (int blkj=0;blkj<rowlen;blkj++)
                {
                    if (bwidth < tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] )
                        bwidth = tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] ;
                }
                double row_length_mean = ((double)tilennz) / rowlen;
                double variance             = 0.0;
                double row_length_skewness   = 0.0;

                for (int row = 0; row < rowlen; ++row)
                {
                    int length              = tile_csr_ptr[tile_id * BLOCK_SIZE + row];
                    double delta                = (double)(length - row_length_mean);
                    variance   += (delta * delta);
                    row_length_skewness   += (delta * delta * delta);
                }
                variance                    /= rowlen;
                double row_length_std_dev    = sqrt(variance);
                row_length_skewness   = (row_length_skewness / rowlen) / pow(row_length_std_dev, 3.0);
                double row_length_variation  = row_length_std_dev / row_length_mean;

                double ell_csr_threshold = 0.2;
                double csr_hyb_threshold = 1.0;

                if (row_length_variation <= ell_csr_threshold)  // if variation is less than 0.2, then ELL
                {
                    Format[tile_id] = 2;
                    blkwidth[tile_id]=bwidth;
                    blknnz[tile_id] = bwidth * rowlen ;
                    ell_offset[tile_id] = bwidth * rowlen;
                }
                else
                {
                    int hybwidth=bwidth;

                    int iopriorsize=  bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (unsigned char)  ;
                                                                // bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (char) /2 +1 ;
                    int ionextsize;
                    int coonextnum=0;
                    int coopriornum=0;
                    for (int wi=bwidth-1;wi>0;wi--)
                    {
                        coonextnum=0;
                        for (int blkj=0;blkj<rowlen;blkj++)
                        {
                            if ( tile_csr_ptr[tile_id * BLOCK_SIZE + blkj]> wi) 
                                {
                                    coonextnum += tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] - wi ;
                                } 
                        }
                        ionextsize=  wi * rowlen * sizeof (MAT_VAL_TYPE )+  wi * rowlen * sizeof (unsigned char)  + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (unsigned char)) ;
                                                                // wi * rowlen * sizeof (MAT_VAL_TYPE )+  wi * rowlen * sizeof (char) /2 + 1 + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (char)) ;
                        if (iopriorsize<=ionextsize)
                        {
                            hybwidth=wi+1;
                            break;
                        }
                        else
                        {
                            hybwidth = wi;
                            iopriorsize=ionextsize;
                            coopriornum=coonextnum;
                        }

                    }
                    
                    if (row_length_variation >= csr_hyb_threshold )//&& coopriornum <= 4)  // if variation > 1.0, and the number of coo data <=4, then HYB
                    {
                        Format[tile_id] = 3;
                        hyb_coocount[tile_id] = coopriornum;
                        blkwidth[tile_id]=hybwidth;
                        blknnz[tile_id] = coopriornum + hybwidth * rowlen ;
                        hyb_offset[tile_id] = coopriornum + hybwidth * rowlen;
                        new_coocount[tile_id] = coopriornum;

                    }
                    else  //else CSR
                    {
                        Format[tile_id] =0 ;
                        blknnz[tile_id] = tilennz ;
                        csr_offset[tile_id] = tilennz;
                        csrptr_offset[tile_id] = BLOCK_SIZE;
                    }
                    
                }
            }
        }
    }
}



void step4_kernel(Beidou_Tile_Matrix *matrix, unsigned char *csr_ptr, int *hyb_coocount, int nnz_temp, int tile_count_temp,
                  int *csr_offset, int *csrptr_offset, int *coo_offset, int *ell_offset, int *hyb_offset, int *dns_offset, int *dnsrow_offset, int *dnscol_offset,
                  MAT_VAL_TYPE *new_coo_value, int *new_coo_colidx, int *new_coo_rowidx, int *new_coocount)


// (int m, int n, int *rowpointer, int *columnidx, MAT_VAL_TYPE *value,
//                   int tilem, int tilen, int numtile, MAT_PTR_TYPE *tile_ptr, int *tile_columnidx, int *tile_nnz, char *Format, 
//                   int *blknnz, unsigned char *csr_ptr,  int nnz_temp,  int tile_count_temp,
//                   MAT_VAL_TYPE *Tile_csr_Val, unsigned char  *Tile_csr_Col, unsigned char *Tile_csr_Ptr, int *csr_offset, int *csrptr_offset, 
//                   MAT_VAL_TYPE *Tile_coo_Val, unsigned char *Tile_coo_colIdx, unsigned char *Tile_coo_rowIdx, int *coo_offset,
//                   MAT_VAL_TYPE *Tile_ell_Val, unsigned char *Tile_ell_colIdx, char *blkwidth, int *ell_offset, 
//                   MAT_VAL_TYPE *Tile_hyb_Val, unsigned char *Tile_hyb_ellcolIdx, unsigned char *Tile_hyb_coorowIdx,  int *hyb_coocount, int *hyb_offset,
//                   MAT_VAL_TYPE *Tile_dns_Val, int *dns_offset,
//                   MAT_VAL_TYPE *Tile_dnsrow_Val, char *Tile_dnsrow_idx, int * denserowptr, int *dnsrow_offset,
//                   MAT_VAL_TYPE *Tile_dnscol_Val, char *Tile_dnscol_idx,  int *densecolptr, int *dnscol_offset){
{
    int *rowpointer=matrix->rowpointer;
    int *columnidx = matrix->columnidx;
    MAT_VAL_TYPE *value = matrix->value;
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int numtile = matrix->numtile;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    int *blknnz = matrix->blknnz;
    char *blkwidth = matrix->blkwidth;
    MAT_VAL_TYPE *Tile_csr_Val = matrix->Tile_csr_Val;
	unsigned char  *Tile_csr_Col = matrix->Tile_csr_Col;
	unsigned char *Tile_csr_Ptr = matrix->Tile_csr_Ptr;
    MAT_VAL_TYPE *Tile_coo_Val = matrix->Tile_coo_Val;
    unsigned char *Tile_coo_colIdx = matrix->Tile_coo_colIdx;
    unsigned char *Tile_coo_rowIdx = matrix->Tile_coo_rowIdx;
    MAT_VAL_TYPE *Tile_ell_Val = matrix->Tile_ell_Val;
    unsigned char *Tile_ell_colIdx = matrix->Tile_ell_colIdx;
    MAT_VAL_TYPE *Tile_hyb_Val = matrix->Tile_hyb_Val;
    unsigned char *Tile_hyb_ellcolIdx = matrix->Tile_hyb_ellcolIdx;
    unsigned char *Tile_hyb_coorowIdx = matrix->Tile_hyb_coorowIdx;
    MAT_VAL_TYPE *Tile_dns_Val = matrix->Tile_dns_Val;
    MAT_VAL_TYPE *Tile_dnsrow_Val = matrix->Tile_dnsrow_Val;
    char *Tile_dnsrow_idx = matrix->Tile_dnsrow_idx;
    MAT_VAL_TYPE *Tile_dnscol_Val = matrix->Tile_dnscol_Val;
    char *Tile_dnscol_idx = matrix->Tile_dnscol_idx;
    int *denserowptr = matrix->denserowptr;
    int *densecolptr = matrix->densecolptr;
    unsigned short *mask = matrix->mask;


    unsigned thread = omp_get_max_threads();
	unsigned char  *csr_colidx_temp_g=(unsigned char*)malloc((thread * nnz_temp )*sizeof(unsigned char));
    MAT_VAL_TYPE *csr_val_temp_g=(MAT_VAL_TYPE*)malloc((thread * nnz_temp)*sizeof(MAT_VAL_TYPE));
    int *tile_count_g = (int *)malloc(thread * tile_count_temp * sizeof(int));

    //for each tile
    #pragma omp parallel for  
    for (int blki=0;blki<tilem;blki++)
	{
       
        int thread_id = omp_get_thread_num();
        unsigned char  *csr_colidx_temp = csr_colidx_temp_g + thread_id * nnz_temp;
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + thread_id * nnz_temp;
        int *tile_count = tile_count_g + thread_id * tile_count_temp;
        // unsigned char  *csr_colidx_temp = (unsigned char *)malloc((nnz_temp )*sizeof(unsigned char));
        // MAT_VAL_TYPE *csr_val_temp = (MAT_VAL_TYPE *)malloc((nnz_temp)*sizeof(MAT_VAL_TYPE));
        // int *tile_count = (int *)malloc(tile_count_temp * sizeof(int));        
        memset(csr_colidx_temp, 0, (nnz_temp)*sizeof(unsigned char));
        memset(csr_val_temp, 0, (nnz_temp)*sizeof(MAT_VAL_TYPE));
        memset(tile_count, 0, (tile_count_temp)*sizeof(int));
        int tilenum_per_row=tile_ptr[blki+1]-tile_ptr[blki];
        int rowlen= blki==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int start = blki*BLOCK_SIZE;
        int end = blki==tilem-1 ?  m : (blki+1)*BLOCK_SIZE ;

        // if (blki == 978)
        // {
        //     printf("thread_id= ,tilenum_per_row=%i, nnz = %i\n", tilenum_per_row, rowpointerA[end]-rowpointerA[start]);
        //     printf("start = %i, end = %i\n",start, end);
        // }
        
        for (int blkj = rowpointer[start]; blkj < rowpointer[end]; blkj ++)
        {
            int jc_temp = columnidx[blkj]/BLOCK_SIZE;
         //   printf("blkj = %i,col=%i\n", blkj, jc_temp);
            for (int bi = 0; bi < tilenum_per_row; bi ++)
            {
                int tile_id = tile_ptr[blki]+bi;
                int jc = tile_columnidx[tile_id];
                int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
                if (jc == jc_temp)
                {
                    csr_val_temp[pre_nnz + tile_count[bi]] = value[blkj];
                    csr_colidx_temp[pre_nnz + tile_count[bi]] = columnidx[blkj] - jc * BLOCK_SIZE;
                //       printf("tile_id = %i, tilennz = %i, jc = %i, prennz = %i, val[%i]=%f,col_before= %i, col[] = %i\n",tile_id, tilennz, jc, pre_nnz,pre_nnz + tile_count[bi],csr_val_temp[pre_nnz + tile_count[bi]], columnindexA[blkj],csr_colidx_temp[pre_nnz + tile_count[bi]]);
                    tile_count[bi] ++;
                    break;
                       
                }
            }
        }

        for (int bi = 0; bi < tilenum_per_row; bi ++)
        {
            int tile_id = tile_ptr[blki]+bi;
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
            int tilennz = tile_nnz[tile_id +1] - tile_nnz[tile_id];  //blknnz[tile_id+1] - blknnz[tile_id] ;
            int collen = tile_columnidx[tile_id] == tilen-1 ? n - (tilen-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            int format = Format[tile_id];
            switch (format)
            {
                case 0:
                {
                    int offset = csr_offset[tile_id];
                    int ptr_offset = csrptr_offset[tile_id];

                    unsigned char *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                    for (int ri =0; ri < rowlen; ri ++)
                    {
                        int start = ptr_temp[ri];
                        int stop = ri == rowlen -1 ? tilennz : ptr_temp[ ri +1];;
                        for (int k =start; k < stop; k ++)
                        {
                            unsigned char colidx = csr_colidx_temp[pre_nnz + k];
                            Tile_csr_Val[offset + k] = csr_val_temp[pre_nnz + k];
                            Tile_csr_Col[offset + k] = csr_colidx_temp[pre_nnz + k];
                            mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));

                        }
                            Tile_csr_Ptr[ptr_offset+ ri] = ptr_temp[ri];

                    }




                    // for (int k = 0; k < tilennz; k++)
                    // {
                    //     Tile_csr_Val[offset + k] = csr_val_temp[pre_nnz + k];
                    //     Tile_csr_Col[offset + k] = csr_colidx_temp[pre_nnz + k];
                    // }
                    // //CSR ptr
                    // for (int pid=0; pid<rowlen; pid++)
                    // {
                    //     Tile_csr_Ptr[ptr_offset+ pid] = ptr_temp[pid];
                    //     mask[tile_id * BLOCK_SIZE + pid] |= (0x1 << (BLOCK_SIZE - colidx - 1));

                    //  //   printf("tile_csr_ptr = %i, csr_ptr = %i\n", Tile_csr_Ptr[ptr_offset+ pid] , csr_ptr[tile_id * BLOCK_SIZE + pid]);
                    // }
                    // unsigned char old_val = Tile_csr_Ptr[ptr_offset];
                    // unsigned char new_val;
                    // Tile_csr_Ptr[ptr_offset] =0;
                    // for (int pid =1; pid < BLOCK_SIZE; pid ++)
                    // {
                    //     new_val = Tile_csr_Ptr[ptr_offset+pid];
                    //     Tile_csr_Ptr[ptr_offset+pid] = old_val + Tile_csr_Ptr[ptr_offset+pid -1];
                    //     old_val = new_val;
                    // }
                    break;
                }

                case 1:
                {
                    if(SPMV && !SPGEMM)
                    {
                        // printf("do spmv operation\n");

                        int colidx_temp = tile_columnidx[tile_id];

                        int offset_new = new_coocount[tile_id];
                        unsigned char *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                        exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                        for (int ri = 0; ri < rowlen; ri++)
                        {
                            int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ ri +1];
                            for (int j = ptr_temp[ri]; j < nnz_end; j++)
                            {
                                new_coo_rowidx[offset_new + j] = ri + blki * BLOCK_SIZE;
                                new_coo_value[offset_new + j] = csr_val_temp[pre_nnz + j] ;
                                new_coo_colidx[offset_new + j]=csr_colidx_temp[pre_nnz + j] + colidx_temp * BLOCK_SIZE;

                            }
                        }

                    }

                    if(SPGEMM && !SPMV)
                    {
                        // printf("do spgemm operation\n");
                        int offset = coo_offset[tile_id];

                        unsigned char *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                        exclusive_scan_char(ptr_temp, rowlen);


                        for (int ri = 0; ri < rowlen; ri++)
                        {
                            int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ ri +1];
                            for (int j = ptr_temp[ ri]; j < nnz_end; j++)
                            {
                                unsigned char colidx = csr_colidx_temp[pre_nnz + j];
                                Tile_coo_rowIdx[offset+ j] = ri;
                                Tile_coo_Val[offset + j] = csr_val_temp[pre_nnz + j] ;
                                Tile_coo_colIdx[offset + j]=csr_colidx_temp[pre_nnz + j];
                                mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                            }
                        }


                    }
                    

                    
                    break;
                 }

                case 2:
                {
                    int offset = ell_offset[tile_id];
                    unsigned char *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_char(ptr_temp, BLOCK_SIZE);
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];

                        for (int j = ptr_temp[ri]; j < nnz_end; j++)
                        {
                            int colidx = csr_colidx_temp[pre_nnz + j];
                            int temp = j - ptr_temp[ri];
                            Tile_ell_colIdx[offset + temp * rowlen + ri] = csr_colidx_temp[pre_nnz + j];
                            Tile_ell_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz + j];
                            // mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                        }
                    }
                    for (int ri =0; ri < rowlen; ri ++)
                    {
                        for (int bi = 0; bi < blkwidth[tile_id]; bi ++)
                        {
                            int colidx = Tile_ell_colIdx[offset + bi * rowlen + ri];
                            mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                        }
                    }
                    
                    break;
                }
                case 3:
                {
                    int colidx_temp = tile_columnidx[tile_id];
                    int offset = hyb_offset[tile_id];
                    unsigned char *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                    int offset_new = new_coocount[tile_id];

                    int coocount=0;
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];
                        int stop= (nnz_end- ptr_temp[ri]) <= blkwidth[tile_id] ? nnz_end : ptr_temp[ri] + blkwidth[tile_id] ;
                            
                        for (int j = ptr_temp[ri]; j < stop; j++)
                        {
                            int colidx = csr_colidx_temp[pre_nnz + j];
                            // printf("row = %i, j = %i, colidx = %i\n",ri,j,colidx);

                            int temp = j - ptr_temp[ri];
                            Tile_hyb_ellcolIdx[offset + temp * rowlen + ri] = csr_colidx_temp[pre_nnz + j];
                            Tile_hyb_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz + j];
                            // mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                            // printf("pos = %i, mask = %i\n", ri, mask[tile_id * BLOCK_SIZE + ri]);
                        }

                        if (SPGEMM && !SPMV)
                        {
                            for (int k=stop; k< nnz_end; k++)
                            {
                                unsigned char colidx = csr_colidx_temp[pre_nnz +k];
                                Tile_hyb_Val[offset + blkwidth[tile_id] * rowlen + coocount] = csr_val_temp[pre_nnz +k];
                                Tile_hyb_ellcolIdx[offset + blkwidth[tile_id] * rowlen + coocount] = csr_colidx_temp[pre_nnz +k];
                                Tile_hyb_coorowIdx[hyb_coocount[tile_id] + coocount] = ri;
                                mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                                coocount++;  
                            }


                        }

                        if(SPMV && !SPGEMM)
                        {
                            for (int k=stop; k< nnz_end; k++)
                            {
                                new_coo_value[offset_new + coocount] = csr_val_temp[pre_nnz +k];
                                new_coo_colidx[offset_new+coocount] = csr_colidx_temp[pre_nnz +k] + colidx_temp * BLOCK_SIZE;
                                new_coo_rowidx[offset_new+coocount] = ri + blki * BLOCK_SIZE;
                                coocount++;  
                            }


                        }
                    }
                    for (int ri =0; ri < rowlen; ri ++)
                    {
                        for (int bi = 0; bi < blkwidth[tile_id]; bi ++)
                        {
                            int colidx = Tile_hyb_ellcolIdx[offset + bi * rowlen + ri];
                            mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                        }
                    }

                    
                    break;
                    
                }
                    
                case 4:
                {
                    int offset = dns_offset[tile_id];
                    unsigned char *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];

                        for (int j = ptr_temp[ri]; j < nnz_end; j++)
                        {
                            unsigned char colidx = csr_colidx_temp[pre_nnz +j];
                            Tile_dns_Val[offset + csr_colidx_temp[pre_nnz + j] * rowlen +ri] = csr_val_temp[pre_nnz + j];
                            // mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                        //    Blockdense_Val[dnsnum[rowblock_ptr[rbi]+bi] + subrowmatrixA[bi].columnindex[j] * rowlength + ri]= subrowmatrixA[bi].value[j];
                        }
                    }
                    for (int ri =0; ri < rowlen; ri ++)
                    {
                        for(int j =0; j < BLOCK_SIZE; j ++)
                        {
                            mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - j - 1));
                        }

                    }
                    
                    
                    break;
                }
                
                case 5:
                {
                    int offset = dnsrow_offset[tile_id];
                    int rowoffset = denserowptr[tile_id];
                    unsigned char *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                    int dnsriid=0;
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];
                        if (nnz_end - ptr_temp[ri] == collen)
                        {
                            // printf("tileid = %i, offset = %i, rowoffset = %i, num = %i\n", tile_id, offset, rowoffset, csr_ptr[tile_id * BLOCK_SIZE + ri]);
                            Tile_dnsrow_idx[rowoffset + dnsriid]=ri;
                            dnsriid ++;
                            for (int j = ptr_temp[ri]; j < nnz_end; j++)
                            {
                                unsigned char colidx = csr_colidx_temp[pre_nnz +j];
                                Tile_dnsrow_Val[offset + j] = csr_val_temp[pre_nnz + j];
                                mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                            }
                        }
                    }
                    break;
                }
                
                case 6:
                {
                    int offset = dnscol_offset[tile_id];
                    int coloffset = densecolptr[tile_id];
                    unsigned char *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_char(ptr_temp, BLOCK_SIZE);

                    // for (int ni =0; ni < rowlen  ; ni ++)
                    // {
                    //     printf("%i    ", ptr_temp[ni]);
                    // }
                    // printf("\n");

                    int dnsciid=0;
                    for (int j=ptr_temp[0];j < ptr_temp[1];j ++)
                    {
                        int ci = csr_colidx_temp[pre_nnz + j];
                        // int ci = subrowmatrixA[bi].columnindex[j] ;
                        Tile_dnscol_idx[coloffset + dnsciid] =ci ;
                //        printf("pos=%i, colidx=%i\n",densecolptr[tile_id + dnsciid],Tile_dnscol_idx[coloffset + dnsciid] );
                        dnsciid++;
                    }
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];

                        for (int j = ptr_temp[ri]; j < nnz_end; j++)
                        {
                            int temp = j - ptr_temp[ri];
                            unsigned char colidx = csr_colidx_temp[pre_nnz +j];//temp;
                            //  if (csr_val_temp[pre_nnz +j] != 0)
                        //    printf("idx = %i, col=%i, val = %f\n",pre_nnz +j, csr_colidx_temp[pre_nnz +j] , csr_val_temp[pre_nnz +j]);
                            Tile_dnscol_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz +j];
                            mask[tile_id * BLOCK_SIZE + ri] |= (0x1 << (BLOCK_SIZE - colidx - 1));

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



void Tile_destroy(Beidou_Tile_Matrix *matrix)
{
    free(matrix->Tile_csr_Col);
    matrix->Tile_csr_Col = NULL;
    free(matrix->Tile_csr_Ptr);
    matrix->Tile_csr_Ptr = NULL;
    free(matrix->Tile_csr_Val);
    matrix->Tile_csr_Val = NULL;
    free(matrix->Tile_coo_colIdx);
    matrix->Tile_coo_colIdx = NULL;
    free(matrix->Tile_coo_rowIdx);
    matrix->Tile_coo_rowIdx = NULL;
    free(matrix->Tile_coo_Val);
    matrix->Tile_coo_Val = NULL;
    free(matrix->Tile_ell_colIdx);
    matrix->Tile_ell_colIdx = NULL;
    free(matrix->Tile_ell_Val);
    matrix->Tile_ell_Val = NULL;
    free(matrix->Tile_hyb_coorowIdx);
    matrix->Tile_hyb_coorowIdx = NULL;
    free(matrix->Tile_hyb_ellcolIdx);
    matrix->Tile_hyb_ellcolIdx = NULL;
    free(matrix->Tile_hyb_Val);
    matrix->Tile_hyb_Val = NULL;
    free(matrix->Tile_dns_Val);
    matrix->Tile_dns_Val = NULL;
    free(matrix->Tile_dnsrow_idx);
    matrix->Tile_dnsrow_idx = NULL;
    free(matrix->Tile_dnsrow_Val);
    matrix->Tile_dnsrow_Val = NULL;
    free(matrix->Tile_dnscol_Val);
    matrix->Tile_dnscol_Val = NULL;
    free(matrix->Tile_dnscol_idx);
    matrix->Tile_dnscol_idx = NULL;
    free(matrix->densecolptr);
    matrix->densecolptr = NULL;
    free(matrix->denserowptr);
    matrix->denserowptr = NULL;
    free(matrix->blkwidth);
    matrix->blkwidth = NULL;
    free(matrix->tile_ptr);
    matrix->tile_ptr = NULL;
    free(matrix->tile_columnidx);
    matrix->tile_columnidx = NULL;
    free(matrix->tile_nnz);
    matrix->tile_nnz = NULL;

    free(matrix->blknnz);
    matrix->blknnz = NULL;
    free(matrix->value);
    matrix->value = NULL;
    free(matrix->columnidx);
    matrix->columnidx = NULL;
    free(matrix->coo_new_matrix_ptr);
    matrix->coo_new_matrix_ptr = NULL;
    free(matrix->coo_new_rowidx);
    matrix->coo_new_rowidx = NULL;
    free(matrix->coo_new_matrix_value);
    matrix->coo_new_matrix_value = NULL;
    free(matrix->coo_new_matrix_colidx);
    matrix->coo_new_matrix_colidx = NULL;

    free(matrix->csr_ptr);
    free(matrix->csr_offset);
    free(matrix->csrptr_offset);
    free(matrix->coo_offset);
    free(matrix->ell_offset);
    free(matrix->hyb_offset);
    free(matrix->dns_offset);
    free(matrix->dnsrow_offset);
    free(matrix->dnscol_offset);





}




void format_transform(Beidou_Tile_Matrix *matrix, 
                  MAT_VAL_TYPE **new_coo_value_temp, int **new_coo_colidx_temp, int **new_coo_rowidx_temp, int **new_coocount_temp)
{

  //unsigned char  *csr_ptr = matrix->csr_ptr;
    // int *csr_offset = matrix->csr_offset;
    // int *csrptr_offset = matrix->csrptr_offset;

    // int *coo_offset = matrix->coo_offset;

    // int *ell_offset = matrix->ell_offset;

    // int *hyb_offset = matrix->hyb_offset;

    // int *dns_offset = matrix->dns_offset;
    // int *dnsrow_offset = matrix->dnsrow_offset;
    // int *dnscol_offset = matrix->dnscol_offset;

    // int *hyb_coocount = matrix->hyb_coocount;

 	struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    // step1_kernel(matrixA->m, matrixA->n, matrixA->rowpointer, matrixA->columnidx, 
    //              matrixA->tilem, matrixA->tilen, matrixA->tile_ptr, &matrixA->numtile);

    step1_kernel(matrix);

    gettimeofday(&t2, NULL);
   // printf("t1=%f,t2=%f\n",t1,t2);
    double time_step1  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("transform_step1 runtime    = %4.5f ms\n", time_step1);

    exclusive_scan(matrix->tile_ptr, matrix->tilem+1);

    matrix->numtile = matrix->tile_ptr[matrix->tilem];
    printf("the number of tiles in matrix A= %d\n",matrix->numtile);

    matrix->tile_columnidx=(int *)malloc(matrix->numtile*sizeof(int));
    memset(matrix->tile_columnidx, 0, matrix->numtile*sizeof(int));

    matrix->tile_nnz = (int *)malloc((matrix->numtile + 1)* sizeof(int)); //real nnz of each sparse tile 
    memset(matrix->tile_nnz,0,(matrix->numtile + 1) * sizeof(int));

    matrix->csr_ptr = (unsigned char *)malloc((matrix->numtile * BLOCK_SIZE) * sizeof(unsigned char));
    memset (matrix->csr_ptr, 0, (matrix->numtile * BLOCK_SIZE) * sizeof(unsigned char));

    gettimeofday(&t1, NULL);
    // step2_kernel_new(matrixA->m, matrixA->n, matrixA->rowpointer, matrixA->columnidx,
    //                  matrixA->tilem, matrixA->tilen, matrixA->tile_ptr, matrixA->tile_columnidx, matrixA->tile_nnz, csr_ptr, matrixA->numtile);

    step2_kernel_new(matrix, matrix->csr_ptr);

    gettimeofday(&t2, NULL);
    double time_step2  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("transform_step2 runtime    = %4.5f ms\n", time_step2);

    exclusive_scan(matrix->tile_nnz, matrix->numtile +1);

       //format 0-7 represent 7 formats: CSR, COO, ELL, HYB, Dns, DnsRow, DnsCol
    matrix->Format =(char *)malloc(matrix->numtile* sizeof(char));
    memset(matrix->Format,0,matrix->numtile * sizeof(char));



    matrix->blknnz = (int *)malloc((matrix->numtile + 1)* sizeof(int)); //space cost that need allocate                                                                                                                    

    memset(matrix->blknnz,0,(matrix->numtile + 1) * sizeof(int));    

    //dense 
    int dense_size=0;
    matrix->dns_offset = (int *)malloc((matrix->numtile+1) * sizeof(int));
    memset(matrix->dns_offset, 0, (matrix->numtile+1) * sizeof(int));


    //denserow
    matrix->denserowptr = (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(matrix->denserowptr,0,(matrix->numtile+ 1) * sizeof(int));
    int denserow_size =0 ;
    matrix->dnsrow_offset = (int *)malloc((matrix->numtile+1) * sizeof(int));
    memset(matrix->dnsrow_offset, 0, (matrix->numtile+1) * sizeof(int));
    //densecolumn
    matrix->densecolptr = (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(matrix->densecolptr,0,(matrix->numtile+ 1) * sizeof(int));
    int densecol_size =0 ;
    matrix->dnscol_offset = (int *)malloc((matrix->numtile+1) * sizeof(int));
    memset(matrix->dnscol_offset, 0, (matrix->numtile+1) * sizeof(int));


    //CSR
    int csrsize=0;
  //  int csrptrlen=0;
    matrix->csr_offset = (int *)malloc((matrix->numtile+1) * sizeof(int));
    memset(matrix->csr_offset, 0, (matrix->numtile+1) * sizeof(int));
    matrix->csrptr_offset = (int *)malloc((matrix->numtile+1) * sizeof(int));
    memset(matrix->csrptr_offset, 0, (matrix->numtile+1) * sizeof(int));

    //ELL
    int ellsize =0;
    matrix->ell_offset = (int *)malloc((matrix->numtile+1) * sizeof(int));
    memset(matrix->ell_offset, 0, (matrix->numtile+1) * sizeof(int));

    //COO
    int coosize =0;
    matrix->coo_offset = (int *)malloc((matrix->numtile+1) * sizeof(int));
    memset(matrix->coo_offset, 0, (matrix->numtile+1) * sizeof(int));

    //HYB
    int hybellsize =0;
    int hybcoosize =0;
    int hybsize =0;
    matrix->blkwidth = (char *)malloc(matrix->numtile*sizeof(char));
    memset(matrix->blkwidth,0,matrix->numtile * sizeof(char)) ;
    matrix->hyb_coocount= (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(matrix->hyb_coocount,0,(matrix->numtile + 1) * sizeof(int)) ;

    matrix->hyb_offset = (int *)malloc((matrix->numtile+1) * sizeof(int));
    memset(matrix->hyb_offset, 0, (matrix->numtile+1) * sizeof(int));

    *new_coocount_temp = (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(*new_coocount_temp,0,(matrix->numtile + 1) * sizeof(int)) ;
    int *new_coocount = *new_coocount_temp;


    gettimeofday(&t1, NULL);

    // step3_kernel_new(matrixA->m, matrixA->n, matrixA->rowpointer, matrixA->columnidx,
    //              matrixA->tilem, matrixA->tilen, matrixA->numtile, matrixA->tile_ptr, matrixA->tile_columnidx, matrixA->tile_nnz, matrixA->Format, 
    //              csr_ptr, matrixA->blknnz, matrixA->blkwidth, hyb_coocount,
    //              matrixA->denserowptr, matrixA->densecolptr,
    //              csr_offset, csrptr_offset, coo_offset, ell_offset, hyb_offset, dns_offset, dnsrow_offset, dnscol_offset);

    step3_kernel_new(matrix, new_coocount);

    gettimeofday(&t2, NULL);
    double time_step3  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("transform_step3 runtime    = %4.5f ms\n", time_step3);

    exclusive_scan(matrix->csr_offset, matrix->numtile +1);
    exclusive_scan(matrix->csrptr_offset, matrix->numtile +1);
    exclusive_scan(matrix->coo_offset, matrix->numtile +1);
    exclusive_scan(matrix->ell_offset, matrix->numtile +1);
    exclusive_scan(matrix->hyb_offset, matrix->numtile +1);
    exclusive_scan(matrix->dns_offset, matrix->numtile +1);
    exclusive_scan(matrix->dnsrow_offset, matrix->numtile +1);
    exclusive_scan(matrix->dnscol_offset, matrix->numtile +1);
    exclusive_scan(matrix->denserowptr,matrix->numtile+1);
    exclusive_scan(matrix->densecolptr,matrix->numtile+1);
    exclusive_scan(matrix->hyb_coocount, matrix->numtile +1);
    hybcoosize = matrix->hyb_coocount[matrix->numtile];

    exclusive_scan(new_coocount, matrix->numtile +1);

    matrix->coocount = new_coocount[ matrix->numtile];

    for (int blki=0;blki<matrix->tilem;blki++)
    {
        int rowlength= blki==matrix->tilem-1 ? matrix->m-(matrix->tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int rowbnum=matrix->tile_ptr[blki+1]-matrix->tile_ptr[blki];
        for (int bi=0;bi<rowbnum;bi++)
        {
            char format= matrix->Format[matrix->tile_ptr[blki]+bi];
            switch (format)
            {
                case 0:    //csr
                    csrsize +=  matrix->blknnz[matrix->tile_ptr[blki]+bi];
            //        csrptrlen += rowlength ;
                    break;
                
                case 1:  //coo
                    coosize += matrix->blknnz[matrix->tile_ptr[blki]+bi];
                    break;
                case 2:  //ell
                    ellsize += matrix->blknnz[matrix->tile_ptr[blki]+bi] ;
                    break;
                case 3: //hyb
                    hybsize += matrix->blknnz[matrix->tile_ptr[blki]+bi];
                    hybellsize += matrix->blkwidth[matrix->tile_ptr[blki]+bi] * rowlength;
                    break;
                case 4:
                    dense_size += matrix->blknnz[matrix->tile_ptr[blki]+bi];
                    break;
                case 5:
                    denserow_size += matrix->blknnz[matrix->tile_ptr[blki]+bi];
                    break;
                case 6:
                    densecol_size += matrix->blknnz[matrix->tile_ptr[blki]+bi];
                    break;
            
                default:
                    break;
            }

        }
    }   

    exclusive_scan(matrix->blknnz,(matrix->numtile+1));

    int *formatnum = (int *)malloc(7 * sizeof(int));
    memset(formatnum,0,7 * sizeof(int));

    for (int j=0;j<7;j++)
    {
        for (int i=0;i<matrix->numtile;i++)
        {
            if (matrix->Format[i]==j)
            {
                formatnum[j]++;
             //   printf("%d   ",Format[i]);
             //   break ;
            }
        }
    }

    for (int j=0;j<7;j++)
    {
        printf("format =%i,count =%i\n",j,formatnum[j]);
    }

    int csrtilecount = formatnum[0];


    int nnz_temp =0;
    int tile_count_temp =0;
    for (int blki =0;blki < matrix->tilem; blki ++)
    {
        int start= blki*BLOCK_SIZE;
        int end = blki==matrix->tilem-1 ?  matrix->m : (blki+1)*BLOCK_SIZE ;
        nnz_temp = nnz_temp < matrix->rowpointer[end] - matrix->rowpointer[start] ? matrix->rowpointer[end] - matrix->rowpointer[start] : nnz_temp;
        tile_count_temp = tile_count_temp < matrix->tile_ptr[blki +1] - matrix->tile_ptr[blki] ? matrix->tile_ptr[blki +1] - matrix->tile_ptr[blki] : tile_count_temp;
    }

         //CSR
    matrix->Tile_csr_Val=(MAT_VAL_TYPE*)malloc((csrsize)*sizeof(MAT_VAL_TYPE));
    memset(matrix->Tile_csr_Val, 0, (csrsize)*sizeof(MAT_VAL_TYPE));
	matrix->Tile_csr_Col=(unsigned char*)malloc((csrsize)*sizeof(unsigned char));
    memset(matrix->Tile_csr_Col, 0, (csrsize)*sizeof(unsigned char));
	matrix->Tile_csr_Ptr=(unsigned char*)malloc((csrtilecount * BLOCK_SIZE)*sizeof(unsigned char));
    memset(matrix->Tile_csr_Ptr, 0, (csrtilecount * BLOCK_SIZE )*sizeof(unsigned char));

    //COO
    matrix->Tile_coo_Val=(MAT_VAL_TYPE*)malloc((coosize)*sizeof(MAT_VAL_TYPE));
    memset(matrix->Tile_coo_Val, 0, (coosize)*sizeof(MAT_VAL_TYPE));
    matrix->Tile_coo_colIdx=(unsigned char*)malloc((coosize)*sizeof(unsigned char));
    memset(matrix->Tile_coo_colIdx, 0, (coosize)*sizeof(unsigned char));
    matrix->Tile_coo_rowIdx=(unsigned char*)malloc((coosize)*sizeof(unsigned char));
    memset(matrix->Tile_coo_rowIdx, 0, (coosize)*sizeof(unsigned char));

     //ELL
    matrix->Tile_ell_Val=(MAT_VAL_TYPE*)malloc((ellsize)*sizeof(MAT_VAL_TYPE));
    memset(matrix->Tile_ell_Val,0,(ellsize)*sizeof(MAT_VAL_TYPE));
    matrix->Tile_ell_colIdx=(unsigned char*)malloc((ellsize)*sizeof(unsigned char));
    memset(matrix->Tile_ell_colIdx, 0, sizeof(unsigned char) * ellsize);

     //HYB
    matrix->Tile_hyb_Val=(MAT_VAL_TYPE*)malloc((hybellsize+hybcoosize)*sizeof(MAT_VAL_TYPE));
    memset(matrix->Tile_hyb_Val,0,(hybellsize+hybcoosize)*sizeof(MAT_VAL_TYPE));
    matrix->Tile_hyb_ellcolIdx=(unsigned char*)malloc((hybellsize+hybcoosize)*sizeof(unsigned char));
    matrix->Tile_hyb_coorowIdx=(unsigned char*)malloc((hybcoosize)*sizeof(unsigned char)) ;
    memset(matrix->Tile_hyb_ellcolIdx, 0, sizeof(unsigned char) * (hybellsize+hybcoosize));
    memset(matrix->Tile_hyb_coorowIdx, 0, sizeof(unsigned char) * hybcoosize);

     //dense
    matrix->Tile_dns_Val=(MAT_VAL_TYPE*)malloc((dense_size)*sizeof(MAT_VAL_TYPE));
    memset(matrix->Tile_dns_Val,0,dense_size * sizeof(MAT_VAL_TYPE));

    //dense row
    matrix->Tile_dnsrow_Val=(MAT_VAL_TYPE*)malloc((denserow_size) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Tile_dnsrow_Val,0,denserow_size * sizeof(MAT_VAL_TYPE));
    matrix->Tile_dnsrow_idx = (char *)malloc(matrix->denserowptr[matrix->numtile] * sizeof(char));
    memset(matrix->Tile_dnsrow_idx, 0, matrix->denserowptr[matrix->numtile] * sizeof(char));

    //dense column
    matrix->Tile_dnscol_Val=(MAT_VAL_TYPE*)malloc((densecol_size) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Tile_dnscol_Val, 0, densecol_size * sizeof(MAT_VAL_TYPE));
    matrix->Tile_dnscol_idx = (char *)malloc(matrix->densecolptr[matrix->numtile] * sizeof(char));
    memset(matrix->Tile_dnscol_idx, 0, matrix->densecolptr[matrix->numtile] * sizeof(char));

    //extract COO to a new matrix

    matrix->coocount = hybcoosize + coosize;

    *new_coo_value_temp = (MAT_VAL_TYPE*)malloc(matrix->coocount *sizeof(MAT_VAL_TYPE));
    memset(*new_coo_value_temp, 0, (matrix->coocount) *sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *new_coo_value = *new_coo_value_temp;

    *new_coo_rowidx_temp = (int *)malloc((hybcoosize+ coosize) *sizeof(int));
    memset(*new_coo_rowidx_temp, 0, (hybcoosize+coosize) *sizeof(int));
    int *new_coo_rowidx = *new_coo_rowidx_temp;

    *new_coo_colidx_temp = (int *)malloc((matrix->coocount) *sizeof(int));
    memset(*new_coo_colidx_temp, 0, (matrix->coocount) *sizeof(int));
    int *new_coo_colidx = *new_coo_colidx_temp;


    //mask
    matrix->mask = (unsigned short *)malloc(matrix->numtile * BLOCK_SIZE * sizeof(unsigned short));
    memset(matrix->mask, 0, matrix->numtile * BLOCK_SIZE * sizeof(unsigned short));

    
    gettimeofday(&t1, NULL);

    // step4_kernel(matrixA->m, matrixA->n, matrixA->rowpointer, matrixA->columnidx, matrixA->value,
    //             matrixA->tilem, matrixA->tilen, matrixA->numtile, matrixA->tile_ptr, matrixA->tile_columnidx, matrixA->tile_nnz, matrixA->Format, 
    //             matrixA->blknnz, csr_ptr, nnz_temp, tile_count_temp,
    //             matrixA->Tile_csr_Val, matrixA->Tile_csr_Col, matrixA->Tile_csr_Ptr, csr_offset, csrptr_offset,
    //             matrixA->Tile_coo_Val, matrixA->Tile_coo_colIdx, matrixA->Tile_coo_rowIdx, coo_offset,
    //             matrixA->Tile_ell_Val, matrixA->Tile_ell_colIdx, matrixA->blkwidth, ell_offset,
    //             matrixA->Tile_hyb_Val, matrixA->Tile_hyb_ellcolIdx, matrixA->Tile_hyb_coorowIdx,  hyb_coocount, hyb_offset,
    //             matrixA->Tile_dns_Val, dns_offset,
    //             matrixA->Tile_dnsrow_Val, matrixA->Tile_dnsrow_idx, matrixA->denserowptr, dnsrow_offset,
    //             matrixA->Tile_dnscol_Val, matrixA->Tile_dnscol_idx,  matrixA->densecolptr, dnscol_offset);

    step4_kernel(matrix, matrix->csr_ptr, matrix->hyb_coocount, nnz_temp, tile_count_temp,
                 matrix->csr_offset, matrix->csrptr_offset, matrix->coo_offset, matrix->ell_offset, matrix->hyb_offset, matrix->dns_offset, matrix->dnsrow_offset, matrix->dnscol_offset,
                new_coo_value,new_coo_colidx, new_coo_rowidx, new_coocount);
    gettimeofday(&t2, NULL);
    double time_step4  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("transform_step4 runtime    = %4.5f ms\n", time_step4);


}

#endif
