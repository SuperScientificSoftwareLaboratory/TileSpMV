#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include <mm_malloc.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <nmmintrin.h>

#include <omp.h>


#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 200
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif

#ifndef DBeidou_Tile_Matrix             
#define DBeidou_Tile_Matrix    
typedef struct 
{
    int m;
    int n;
    int nnz;
    int isSymmetric;
    int tilem;
    int tilen;
    int numtile;
    int coocount;
	MAT_VAL_TYPE *value;
	int *columnidx;
	MAT_PTR_TYPE *rowpointer;
    int *tile_ptr;
    int *tile_columnidx;
    int *tile_nnz;
    char *Format;
    int *blknnz;
    unsigned char *csr_ptr ;
    int *csr_ptr_1 ;
    MAT_VAL_TYPE *Tile_csr_Val;
	unsigned char  *Tile_csr_Col;
	unsigned char *Tile_csr_Ptr;
    MAT_VAL_TYPE *Tile_coo_Val;
    unsigned char *Tile_coo_colIdx;
    unsigned char *Tile_coo_rowIdx;
    MAT_VAL_TYPE *Tile_ell_Val;
    unsigned char *Tile_ell_colIdx;
    char *blkwidth;
    MAT_VAL_TYPE *Tile_hyb_Val;
    unsigned char *Tile_hyb_ellcolIdx;
    unsigned char *Tile_hyb_coorowIdx;
    MAT_VAL_TYPE *Tile_dns_Val;
    MAT_VAL_TYPE *Tile_dnsrow_Val;
    char *Tile_dnsrow_idx;
    MAT_VAL_TYPE *Tile_dnscol_Val;
    char *Tile_dnscol_idx;
    int *denserowptr;
    int *densecolptr;
    unsigned int *flag_bal_tile_rowidx;
    int *tile_bal_rowidx_colstart ;
    int *tile_bal_rowidx_colstop ;
    int *coo_new_rowidx ;
    int *coo_new_matrix_ptr ;
    int *coo_new_matrix_colidx ;
    MAT_VAL_TYPE *coo_new_matrix_value ;
   
    int *dns_offset ;
    
    int *dnsrow_offset;
    int *dnscol_offset ;
    int *csr_offset ;
    int *csrptr_offset ;
    int *ell_offset ;
    int *coo_offset ;
    int *hyb_coocount;
    int *hyb_offset ;

    int *Yid;
    int *csrSplitter_yid;
    int *label;
    int *Start1;
    int *End1;
    int *Start2;
    int *End2;

    // int *new_coocount ;
    // MAT_VAL_TYPE *new_coo_value ;
    // int *new_coo_rowidx ;
    // int *new_coo_colidx ;
    unsigned short *mask;
    int csrsize;
    int hybcoosize;
    int coosize;
    int ellsize;
    int hybsize;
    int hybellsize;
    int dense_size;
    int denserow_size;
    int densecol_size;
    int csrtilecount;

   unsigned char *blknnznnz;
    

    
}Beidou_Tile_Matrix;
#endif



// #ifndef BLOCK_SIZE
// #define BLOCK_SIZE  16
// #endif

#ifndef NTHREADS_MAX
#define NTHREADS_MAX 12
#endif

#ifndef COO_THRESHOLD
#define COO_THRESHOLD 12
#endif


// #ifndef PREFETCH_SMEM_TH
// #define PREFETCH_SMEM_TH 8
// #endif
#ifndef SPMV
#define SPMV 1
#endif

#ifndef SPGEMM
#define SPGEMM 0
#endif



#define METHOD_SERIAL               1
#define METHOD_LOCK                 2
#define METHOD_LOCK_AND_DEPENDENCY  3

#define HASH_SCALE 107



# define INDEX_DATA_TYPE unsigned char
//# define VAL_DATA_TYPE double

#define WARMUP_NUM 200

#define WARP_SIZE 32
#define WARP_PER_BLOCK 2

#define num_f 240
#define num_b 15

#define PREFETCH_SMEM_TH 8
#define COO_NNZ_TH 12

#define DEBUG_FORMATCOST 0

#define BLOCK_SIZE 16 
