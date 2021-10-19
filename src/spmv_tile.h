#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"
#include"tilespmv_warp.h"
//#include"tilespmv_warp_avx.h"


void tilespmv(Beidou_Tile_Matrix *matrix, 
              MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int new_row)


// (int rowA, int colA, int *rowpointerA, int *columnindexA, MAT_VAL_TYPE *valueA,
//                   int tilemA, int tilen, int numtileA, MAT_PTR_TYPE *tile_ptr_A, int *tile_columnidx, int *tile_nnz, char *Format, 
//                   int *blknnz, unsigned char *csr_ptr,  int nnz_temp,
//                   MAT_VAL_TYPE *Tile_csr_Val, unsigned char  *Tile_csr_Col, unsigned char *Tile_csr_Ptr, int *csr_offset, int *csrptr_offset, 
//                   MAT_VAL_TYPE *Tile_coo_Val, unsigned char *Tile_coo_colIdx, unsigned char *Tile_coo_rowIdx, int *coo_offset,
//                   MAT_VAL_TYPE *Tile_ell_Val, unsigned char *Tile_ell_colIdx, char *blkwidth, int *ell_offset, 
//                   MAT_VAL_TYPE *Tile_hyb_Val, unsigned char *Tile_hyb_ellcolIdx, unsigned char *Tile_hyb_coorowIdx,  int *hyb_coocount, int *hyb_offset,
//                   MAT_VAL_TYPE *Tile_dns_Val, int *dns_offset,
//                   MAT_VAL_TYPE *Tile_denserow_Val, char *Tile_dnsrow_idx, int * denserowptr, int *dnsrow_offset,
//                   MAT_VAL_TYPE *Tile_dnscol_Val, char *Tile_dnscol_idx,  int *densecolptr, int *dnscol_offset,
//                   MAT_VAL_TYPE *x, MAT_VAL_TYPE *y)
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



    #pragma omp parallel for  
    for (int blki = 0; blki < tilem; blki ++)
    {
        int tilenum_per_row=tile_ptr[blki+1]-tile_ptr[blki];
        int rowlen= blki==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int start = blki*BLOCK_SIZE;
        int end = blki==tilem-1 ?  m : (blki+1)*BLOCK_SIZE ;
        for (int blkj = tile_ptr[blki]; blkj < tile_ptr[blki + 1]; blkj ++)
        {
            int collen = tile_columnidx[blkj] == tilen-1 ? n - (tilen-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            int tilennz = tile_nnz[blkj +1] - tile_nnz[blkj];
            char format = Format[blkj];
            int x_offset = tile_columnidx[blkj] * BLOCK_SIZE;

            switch (format)
            {
            case 0:
            {
                // warplevel_csr(m, n, tilem, tilen, 
                //              tile_ptr, tile_columnidx, tile_nnz, blki, blkj, 
                //              Tile_csr_Val, Tile_csr_Col, Tile_csr_Ptr,csr_offset, csrptr_offset,
                //              x,y, x_offset);
                warplevel_csr(matrix, blki,  blkj, csr_offset, csrptr_offset, 
                              x, y,  x_offset);
               
                break;
            }

            case 1:
            {
                // warplevel_coo(m, n, tilem, tilen, 
                //              tile_ptr, tile_columnidx, tile_nnz, blki, blkj, 
                //              Tile_coo_Val, Tile_coo_colIdx, Tile_coo_rowIdx, coo_offset,
                //              x,y, x_offset);

                // warplevel_coo(matrix, blki, blkj, coo_offset,  
                //               x, y,  x_offset, BLOCK_SIZE);
                break;
            }
            case 2:
            {
                // warplevel_ell(m, n, tilem, tilen, 
                //              tile_ptr, tile_columnidx, tile_nnz, blki, blkj, 
                //              Tile_ell_Val, Tile_ell_colIdx, blkwidth, ell_offset,
                //              x,y, x_offset);
                warplevel_ell(matrix, blki, blkj, ell_offset,  
                              x, y,  x_offset);
                break;
            }
            
            case 3:
            {
                // warplevel_hyb(m, n, tilem, tilen, blknnz,
                //              tile_ptr, tile_columnidx, tile_nnz, blki, blkj, blkwidth,
                //              Tile_hyb_Val, Tile_hyb_ellcolIdx, Tile_hyb_coorowIdx,  hyb_coocount, hyb_offset,
                //              x,y, x_offset);
                warplevel_hyb(matrix, blki, blkj,hyb_coocount, hyb_offset,  
                              x, y,  x_offset);
                
                break;
            }

            case 4:
            {
                // warplevel_dns(m, n, tilem, tilen, 
                //              tile_ptr, tile_columnidx, tile_nnz, blki, blkj, 
                //              Tile_dns_Val,  dns_offset,
                //              x,y, x_offset);
                warplevel_dns(matrix, blki, blkj, dns_offset,  
                              x, y,  x_offset);

               
                break;
            }
           

            case 5:
            {
                // warplevel_dnsrow(m, n, tilem, tilen, 
                //              tile_ptr, tile_columnidx, tile_nnz, blki, blkj, 
                //              Tile_dnsrow_Val, Tile_dnsrow_idx, denserowptr, dnsrow_offset,
                //              x,y, x_offset);
                warplevel_dnsrow(matrix, blki, blkj, dnsrow_offset,  
                                 x, y,  x_offset);
                break;
            }
            

            case 6:
            {
                // warplevel_dnscol(m, n, tilem, tilen, 
                //                 tile_ptr, tile_columnidx, tile_nnz, blki, blkj, 
                //                 Tile_dnscol_Val, Tile_dnscol_idx, densecolptr, dnscol_offset,
                //                 x,y, x_offset);

                warplevel_dnscol(matrix, blki, blkj, dnscol_offset,  
                                 x, y,  x_offset);

                break;
            }
            
            default:
                break;
            }

        }

    }

    #pragma omp parallel for
    for (int ri = 0; ri <new_row; ri++)
    {
        int rowidx = matrix->coo_new_rowidx[ri];
        MAT_VAL_TYPE sum = 0;
        // for each nonzero in the row of the block
        // the last row uses nnzlocal
        for (int rj = matrix->coo_new_matrix_ptr[ri]; rj < matrix->coo_new_matrix_ptr[ri +1]; rj++)
        {
            int csrcolidx = matrix->coo_new_matrix_colidx[rj];
            sum += x[csrcolidx] * matrix->coo_new_matrix_value[rj];

        }
        y[rowidx] += sum;
    }


}
