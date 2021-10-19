#ifndef SPMV_TILE_BALANCE
#define SPMV_TILE_BALANCE

#include"common.h"
// #include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"
#include"tilespmv_warp_bal.h"


void tilespmv_balance(Beidou_Tile_Matrix *matrix, int rowblkblock, 
                    MAT_VAL_TYPE *x, MAT_VAL_TYPE *y_bal,
                    int *flag_tilerow_start, int *flag_tilerow_stop,  MAT_VAL_TYPE *Ysum, MAT_VAL_TYPE *Ypartialsum)

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
    unsigned int *flag_bal_tile_rowidx = matrix->flag_bal_tile_rowidx;
    int *tile_bal_rowidx_colstart = matrix->tile_bal_rowidx_colstart ;
    int *tile_bal_rowidx_colstop = matrix->tile_bal_rowidx_colstop;


    unsigned char *csr_ptr = matrix->csr_ptr;
    int *hyb_coocount = matrix->hyb_coocount;
    int *csr_offset = matrix->csr_offset; 
    int *csrptr_offset = matrix->csrptr_offset;
    int *coo_offset = matrix->coo_offset;
    int *ell_offset = matrix->ell_offset;
    int *hyb_offset = matrix->hyb_offset;
    int *dns_offset = matrix->dns_offset;
    int *dnsrow_offset = matrix->dnsrow_offset;
    int *dnscol_offset = matrix->dnscol_offset;


    int nthreads = omp_get_max_threads();
    MAT_VAL_TYPE *y_temp_g = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * BLOCK_SIZE * nthreads);
    memset(y_temp_g, 0, sizeof(MAT_VAL_TYPE) * BLOCK_SIZE * nthreads);
    int *flag_lastgroup_rowidx = (int *)malloc(nthreads * sizeof(int));
    memset(flag_lastgroup_rowidx, 0, nthreads * sizeof(int));

    // int *flag_tilerow_start = (int *)malloc(nthreads * sizeof(int));
    // memset(flag_tilerow_start, 0, nthreads * sizeof(int));
    // int *flag_tilerow_end = (int *)malloc(nthreads * sizeof(int));
    // memset(flag_tilerow_end, 0, nthreads * sizeof(int));


    // printf("balance rowblkblock = %i\n",rowblkblock);


    // int rowblk_ave = rowblkblock /  nthreads;
    // int rowblk_ave_rest = rowblkblock % nthreads ;

    

    // // printf("tile group_ave = %i\n",rowblk_ave);

    // #pragma omp parallel for
    // for (int i =0; i < nthreads; i ++)
    // {
    //     if (i < rowblk_ave_rest)
    //     {
    //         flag_tilerow_start[i] = i * (rowblk_ave +1);
    //         flag_tilerow_stop[i] = (i +1) * (rowblk_ave +1);
    //     }
    //     else{
    //         flag_tilerow_start[i] = (rowblk_ave +1) * rowblk_ave_rest + (i - rowblk_ave_rest) * rowblk_ave;
    //         flag_tilerow_stop[i] = (rowblk_ave +1) * rowblk_ave_rest + (i +1 - rowblk_ave_rest) * rowblk_ave;
    //     }

     
    // }

            // }
//printf("aaaaaaa0\n");

    #pragma omp parallel for
    for (int ti =0; ti < nthreads; ti ++)
    {
        int start_groupid = flag_tilerow_start[ti];
        int end_groupid = flag_tilerow_stop[ti];

        int thread_id = omp_get_thread_num();

        // if (ti ==nthreads -1)
        // {
        //     printf("thread  %i, start = %i, stop = %i\n",thread_id, start_groupid,end_groupid );
        // }

//printf("aaaaaaa1\n");
        MAT_VAL_TYPE *y_local = (MAT_VAL_TYPE *)malloc (BLOCK_SIZE * sizeof(MAT_VAL_TYPE));
        memset(y_local, 0, BLOCK_SIZE * sizeof(MAT_VAL_TYPE));
//printf("start_groupid=%d  end_groupid=%d\n",start_groupid,end_groupid);
        for (int blki = start_groupid; blki < end_groupid ; blki ++)
        {
            int tile_rowidx_current = flag_bal_tile_rowidx[blki];
            int tile_rowidx_next =  blki == rowblkblock -1 ? -1: flag_bal_tile_rowidx[blki +1];
            int rowlen= tile_rowidx_current==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;

            // if (blki == end_groupid -1)
            // {
            //     printf("tile_rowidx_current = %i,tile_rowidx_next = %i, rowlen= %i\n", tile_rowidx_current, tile_rowidx_next, rowlen);
            // }
//printf("aaaaaaa2\n");
            for (int blkj = tile_bal_rowidx_colstart[blki]; blkj < tile_bal_rowidx_colstop[blki]; blkj ++)
            {
                int collen = tile_columnidx[blkj] == tilen-1 ? n - (tilen-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
                int tilennz = tile_nnz[blkj +1] - tile_nnz[blkj];
                char format = Format[blkj];
                int x_offset = tile_columnidx[blkj] * BLOCK_SIZE;
                //printf("format=%d\n",format);
                switch (format)
                {
                    case 0:
                    {

                        warplevel_csr_bal(matrix, tile_rowidx_current,  blkj, csr_offset, csrptr_offset, 
                                    x,  y_local, x_offset);
                    
                        break;
                    }

                    case 1:
                    {
                        // warplevel_coo_bal(matrix, tile_rowidx_current, blkj, coo_offset,  
                        //             x,  y_local, x_offset);

                        break;
                    }
                    case 2:
                    {
                        warplevel_ell_bal(matrix, tile_rowidx_current, blkj, ell_offset,  
                                    x, y_local,  x_offset);

                        break;
                    }
                
                    case 3:
                    {
                        warplevel_hyb_bal(matrix, tile_rowidx_current, blkj,hyb_coocount, hyb_offset,  
                                    x, y_local,  x_offset);
                        
                        break;
                    }


                    case 4:
                    {
                        warplevel_dns_bal(matrix, tile_rowidx_current, blkj, dns_offset,  
                                    x, y_local,  x_offset);


                    
                        break;
                    }
            

                    case 5:
                    {
                        warplevel_dnsrow_bal(matrix, tile_rowidx_current, blkj, dnsrow_offset,  
                                        x, y_local,  x_offset);
                    
                        break;
                    }
                

                    case 6:
                    {

                        warplevel_dnscol_bal(matrix, tile_rowidx_current, blkj, dnscol_offset,  
                                        x, y_local,  x_offset);

                        break;
                    }
                
                    default:
                        break;
                }

            }

//printf("aaaaaaa2\n");
            if (blki == end_groupid - 1)
            {
                if(tile_rowidx_current != tile_rowidx_next )
                {
                    for (int ri =0; ri < rowlen; ri ++)
                    {
                        y_bal[tile_rowidx_current * BLOCK_SIZE +ri] = y_local[ri];
                    }
                }
                else
                {
                    flag_lastgroup_rowidx[thread_id] = tile_rowidx_current;
                    for (int ri =0; ri < rowlen; ri ++)
                    {
                        y_temp_g[thread_id * BLOCK_SIZE + ri] = y_local[ri];
                    }
                memset(y_local, 0, BLOCK_SIZE * sizeof(MAT_VAL_TYPE));


                }


            }

            else
            {
                if(tile_rowidx_current != tile_rowidx_next)
                {   
                    for (int ri =0; ri < rowlen; ri ++)
                    {
                        y_bal[tile_rowidx_current * BLOCK_SIZE +ri] = y_local[ri];
                    }
                    memset(y_local, 0, BLOCK_SIZE * sizeof(MAT_VAL_TYPE));
                }

            }

          

        }


    }
//printf("aaaaaaa3\n");
    for (int ti =0; ti < nthreads; ti ++)
    {
        int rowidx_temp = flag_lastgroup_rowidx[ti];
        int rowlen = rowidx_temp == tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
        for (int ri =0; ri < rowlen; ri ++)
        {
            y_bal[rowidx_temp * BLOCK_SIZE + ri] += y_temp_g[ti * BLOCK_SIZE +ri];

        }
    }
//printf("aaaaaaa4\n");
    #pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++)
    {
        if (matrix->Yid[tid] == -1 )
        {
            for (int u = matrix->csrSplitter_yid[tid]; u < matrix->csrSplitter_yid[tid+1]; u++)
            { //printf("u=%d\n",u);
               int rowidx = matrix->coo_new_rowidx[u];
               
                double sum = 0;
                 for (int j = matrix->coo_new_matrix_ptr[u]; j < matrix->coo_new_matrix_ptr[u + 1]; j++) 
                {
                    int csrcolidx = matrix->coo_new_matrix_colidx[j];
                    sum += matrix->coo_new_matrix_value[j] * x[csrcolidx];
                }
                y_bal[rowidx] += sum;
            }
        }
        else if (matrix->label[tid] != 0)
        {
            for (int u = matrix->Start1[tid]; u < matrix->End1[tid]; u++)
            {
                int rowidx = matrix->coo_new_rowidx[u];
                double sum = 0;
                for (int j = matrix->coo_new_matrix_ptr[u]; j < matrix->coo_new_matrix_ptr[u + 1]; j++) 
                {
                    int csrcolidx = matrix->coo_new_matrix_colidx[j];
                    sum += matrix->coo_new_matrix_value[j] * x[csrcolidx];
                }
                y_bal[rowidx] += sum;
            }
        }
        else if (matrix->Yid[tid] != -1 && matrix->label[tid] == 0)//youwenti
        {
            Ysum[tid] = 0;
            Ypartialsum[tid] = 0;
            for (int j = matrix->Start2[tid]; j < matrix->End2[tid]; j++)
            {
                int csrcolidx = matrix->coo_new_matrix_colidx[j];
                Ypartialsum[tid] += matrix->coo_new_matrix_value[j] * x[csrcolidx];
            }
            Ysum[tid] += Ypartialsum[tid];
            y_bal[matrix->Yid[tid]] += Ysum[tid];
        }
    }


    
}
#endif

   
