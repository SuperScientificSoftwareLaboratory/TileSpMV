#ifndef TILESPMV_WARP_BAL
#define TILESPMV_WARP_BAL
#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"

void warplevel_csr_bal(Beidou_Tile_Matrix *matrix, int tile_row, int tile_id, int *csr_offset, int *csrptr_offset, 
                   MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)

// (int rowA, int colA, int tilemA, int tilenA, 
//                    int *tile_ptr_A, int *tile_columnidx, int *tile_nnz, int tile_row,  int tile_id, 
//                    MAT_VAL_TYPE *Tile_csr_Val, unsigned char  *Tile_csr_Col, unsigned char *Tile_csr_Ptr, int *csr_offset, int *csrptr_offset,
//                    MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)

{
   
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;

    MAT_VAL_TYPE *Tile_csr_Val = matrix->Tile_csr_Val;
	unsigned char  *Tile_csr_Col = matrix->Tile_csr_Col;
	unsigned char *Tile_csr_Ptr = matrix->Tile_csr_Ptr;

    int tilennz = tile_nnz[tile_id+1] - tile_nnz[tile_id];
    int offset = csr_offset[tile_id];
    int ptr_offset = csrptr_offset[tile_id];
    int rowlen= tile_row==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;


    for (int ri = 0; ri < rowlen; ri++)
    {
        MAT_VAL_TYPE sum = 0;
        // for each nonzero in the row of the block
        // the last row uses nnzlocal
        int stop = ri == rowlen - 1 ? tilennz : Tile_csr_Ptr[ptr_offset + ri +1];
        for (int rj = Tile_csr_Ptr[ptr_offset + ri]; rj < stop; rj++)
        {
            // int csrcol = (csroffset + rj) % 2 ==0 ? (csr_compressedIdx[(csroffset + rj) / 2] & num_f )>> 4 : 
            //                                         csr_compressedIdx[(csroffset + rj) / 2 ] & num_b ;
            int csrcolidx = Tile_csr_Col[offset + rj];
            sum += x[x_offset + csrcolidx] * Tile_csr_Val[offset+rj];

        }
        y[ri] += sum;
    }

}

void warplevel_coo_bal(Beidou_Tile_Matrix *matrix, int tile_row, int tile_id, int *coo_offset, 
                   MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)

// (int rowA, int colA, int tilemA, int tilenA, 
//                    int *tile_pre_A, int *tile_columnidx, int *tile_nnz, int tile_row,  int tile_id, 
//                    MAT_VAL_TYPE *Tile_coo_Val, unsigned char *Tile_coo_colIdx, unsigned char *Tile_coo_rowIdx, int *coo_offset,
//                    MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)
{

    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;

    MAT_VAL_TYPE *Tile_coo_Val = matrix->Tile_coo_Val;
	unsigned char  *Tile_coo_rowIdx = matrix->Tile_coo_rowIdx;
	unsigned char *Tile_coo_colIdx = matrix->Tile_coo_colIdx;

    int tilennz = tile_nnz[tile_id+1] - tile_nnz[tile_id];
    int offset = coo_offset[tile_id];


    for (int rj= 0; rj < tilennz; rj++)
    {
        unsigned char coorowidx = Tile_coo_rowIdx[offset + rj];
        unsigned char coocolidx = Tile_coo_colIdx[offset + rj];
        y[coorowidx] += Tile_coo_Val[offset + rj] * x[x_offset + coocolidx]; 
    }
    
}

void warplevel_ell_bal(Beidou_Tile_Matrix *matrix, int tile_row, int tile_id, int *ell_offset, 
                   MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)

// (int rowA, int colA, int tilemA, int tilenA, 
//                    int *tile_pre_A, int *tile_columnidx, int *tile_nnz, int tile_row,  int tile_id, 
//                    MAT_VAL_TYPE *Tile_ell_Val, unsigned char *Tile_ell_colIdx, char *blkwidth, int *ell_offset, 
//                    MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)
{
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;

    MAT_VAL_TYPE *Tile_ell_Val = matrix->Tile_ell_Val;
	unsigned char  *Tile_ell_colIdx = matrix->Tile_ell_colIdx;
	char *blkwidth = matrix->blkwidth;
    int offset = ell_offset[tile_id];
    int rowlen= tile_row==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
//    for each row in the block
    for (int ri = 0; ri < rowlen; ri++)
    {
        MAT_VAL_TYPE sum = 0;
        // for each nonzero in the row of the block
        // the last row uses nnzlocal
            for (int j = 0; j < blkwidth[tile_id]; j++)
            {
                // int ellcol = (ellwoffset+ j * rowlength + ri) % 2 ==0 ? (ell_compressedIdx[(ellwoffset+ j * rowlength + ri) / 2] & num_f )>> 4 : 
                //                                                          ell_compressedIdx[(ellwoffset+ j * rowlength + ri) / 2 ] & num_b ;
                int ellcolidx = Tile_ell_colIdx[offset + j * rowlen + ri];
                sum += Tile_ell_Val[offset+j * rowlen + ri] * x[x_offset+ ellcolidx];  
            }
            y[ri] += sum;                        
    }

    
}

void warplevel_hyb_bal(Beidou_Tile_Matrix *matrix, int tile_row, int tile_id, int *hyb_coocount, int *hyb_offset, 
                   MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)

// (int rowA, int colA, int tilemA, int tilenA, int *blknnz,
//                    int *tile_pre_A, int *tile_columnidx, int *tile_nnz, int tile_row,  int tile_id, char *blkwidth, 
//                    MAT_VAL_TYPE *Tile_hyb_Val, unsigned char *Tile_hyb_ellcolIdx, unsigned char *Tile_hyb_coorowIdx,  int *hyb_coocount, int *hyb_offset,
//                    MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)
{
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    int *blknnz = matrix->blknnz;

    MAT_VAL_TYPE *Tile_hyb_Val = matrix->Tile_hyb_Val;
	unsigned char  *Tile_hyb_ellcolIdx = matrix->Tile_hyb_ellcolIdx;
    unsigned char *Tile_hyb_coorowIdx = matrix->Tile_hyb_coorowIdx;

	char *blkwidth = matrix->blkwidth;




    int offset = hyb_offset[tile_id];
    int offset_coo = hyb_coocount[tile_id];
        int rowlen= tile_row==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;

    for (int ri = 0; ri < rowlen; ri++)
    {
        MAT_VAL_TYPE sum = 0;
        for (int j = 0; j < blkwidth[tile_id]; j++)
        {
            unsigned char hybcolidx = Tile_hyb_ellcolIdx[offset + j * rowlen + ri];
            sum += Tile_hyb_Val[offset + j * rowlen + ri] * x[x_offset + hybcolidx];
        }
        y[ri] += sum;

    }
    // int coonnz = blknnz[tile_id +1] - blknnz[tile_id] - blkwidth[tile_id] * rowlen;
    // for (int i = 0; i < coonnz; i++)
    // {
    //     unsigned char hybrowidx = Tile_hyb_coorowIdx[offset_coo + i];
    //     unsigned char hyb_coo_colidx = Tile_hyb_ellcolIdx[offset + blkwidth[tile_id]* rowlen + i];
    //     y[hybrowidx] += Tile_hyb_Val[offset + blkwidth[tile_id]* rowlen + i] * x[x_offset + hyb_coo_colidx];
    // }
    
}

void warplevel_dns_bal(Beidou_Tile_Matrix *matrix, int tile_row, int tile_id, int *dns_offset, 
                   MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)
// (int rowA, int colA, int tilemA, int tilenA, 
//                    int *tile_pre_A, int *tile_columnidx, int *tile_nnz, int tile_row,  int tile_id, 
//                    MAT_VAL_TYPE *Tile_dns_Val, int *dns_offset, 
//                    MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)
{

    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    int *blknnz = matrix->blknnz;

    MAT_VAL_TYPE *Tile_dns_Val = matrix->Tile_dns_Val;


    int offset = dns_offset[tile_id];
    int rowlen= tile_row==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
    int collen = tile_columnidx[tile_id] == tilen-1 ? n - (tilen-1 ) * BLOCK_SIZE : BLOCK_SIZE ;

    for (int ri = 0; ri < rowlen; ri++)
    {
        MAT_VAL_TYPE sum = 0;
        // for each nonzero in the row of the block
        // the last row uses nnzlocal
    //	int stop = ri == BLOCK_SIZE - 1 ? (nnzb_A[blkj+1]-nnzb_A[blkj]) : BlockA_Ptr[ri+1+blkj*BLOCK_SIZE];
        for (int rj = ri * collen; rj < (ri +1)*collen; rj++)
        {
            int densecolidx=rj % collen ;
            y[ri] += x[x_offset + densecolidx] * Tile_dns_Val[offset +densecolidx * rowlen + ri]; //Blockdense_Val[denseoffset +rj];
        }
        //y[blki * BLOCK_SIZE + ri] += sum;
    }
    
}

void warplevel_dnsrow_bal(Beidou_Tile_Matrix *matrix, int tile_row, int tile_id, int *dnsrow_offset, 
                      MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)

// (int rowA, int colA, int tilemA, int tilenA, 
//                    int *tile_pre_A, int *tile_columnidx, int *tile_nnz, int tile_row,  int tile_id, 
//                    MAT_VAL_TYPE *Tile_denserow_Val, char *Tile_dnsrow_idx, int * denserowptr, int *dnsrow_offset,
//                    MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)
{
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    int *blknnz = matrix->blknnz;

    MAT_VAL_TYPE *Tile_dnsrow_Val = matrix->Tile_dnsrow_Val;
    int *denserowptr = matrix->denserowptr;
    char *Tile_dnsrow_idx = matrix->Tile_dnsrow_idx;


    int offset = dnsrow_offset[tile_id];
    int collen = tile_columnidx[tile_id] == tilen-1 ? n - (tilen-1 ) * BLOCK_SIZE : BLOCK_SIZE ;

    for (int ri=denserowptr[tile_id]; ri < denserowptr[ tile_id +1 ];ri++)
    {
        MAT_VAL_TYPE sum = 0;
        for (int rj = 0; rj < collen; rj++)
        {
            sum += x[x_offset + rj] * Tile_dnsrow_Val[offset + (ri - denserowptr[tile_id]) * collen +rj];
        }
        y[Tile_dnsrow_idx[ri]] += sum;
    }
    
}

void warplevel_dnscol_bal
(Beidou_Tile_Matrix *matrix, int tile_row, int tile_id, int *dnscol_offset, 
                      MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)

// (int m, int n, int tilem, int tilen, 
//                    int *tile_pre, int *tile_columnidx, int *tile_nnz, int tile_row,  int tile_id, 
//                    MAT_VAL_TYPE *Tile_dnscol_Val, char *Tile_dnscol_idx,  int *densecolptr, int *dnscol_offset,
//                    MAT_VAL_TYPE *x, MAT_VAL_TYPE *y, int x_offset)
{

    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    int *blknnz = matrix->blknnz;

    MAT_VAL_TYPE *Tile_dnscol_Val = matrix->Tile_dnscol_Val;
    int *densecolptr = matrix->densecolptr;
    char *Tile_dnscol_idx = matrix->Tile_dnscol_idx;

    int rowlen= tile_row==tilem-1 ? m-(tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
    int offset = dnscol_offset[tile_id];
    for (int ri=0 ; ri < rowlen;ri++)
    {
        MAT_VAL_TYPE sum = 0;
        for (int rj = densecolptr[tile_id]; rj < densecolptr[tile_id +1]; rj++)
        {
            sum += Tile_dnscol_Val[offset + (rj-densecolptr[tile_id]) * rowlen + ri] * x[x_offset+ Tile_dnscol_idx[rj]];  
        }
        y[ri] += sum;
    }

    
}
#endif
