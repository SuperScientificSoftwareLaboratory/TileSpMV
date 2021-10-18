#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"
#include"utils_tile.h"
#include"format_trans.h"
#include"spmv_tile.h"
#include"spmv_tile_balance.h"
#include"tilespmv_warp_bal.h"
//#include"tilespmv_warp_avx.h"
//#include"spmv_tile_balance_avx.h"
#include"LBLT.h"
#include"step.h"
#include"spmv_cuda.h"
#include <thrust/sort.h>


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

int main(int argc, char ** argv)
{

   
printf("--------------------------------!!-cuda-!!------------------------------------\n");
        Beidou_Tile_Matrix *matrixA_d = (Beidou_Tile_Matrix *)malloc(sizeof(Beidou_Tile_Matrix));
	//SMatrix matrixA_1;
    struct timeval t1, t2;
    int argi = 1;
 //   int BLOCK_SIZE;
 /*   if(argc > argi)
    {
        BLOCK_SIZE = atoi(argv[argi])  ;
        argi++;
    }

    printf(" -------------- BLOCK SIZE = %i --------------\n", BLOCK_SIZE);*/


    int nthreads;
    if(argc > argi)
    {
        nthreads = atoi(argv[argi]);
        argi++;
    }
omp_set_num_threads(nthreads);
    printf(" -------------- threads = %i --------------\n", nthreads);

    char  *filename;
    filename = argv[2];
    printf("MAT: -------------- %s --------------\n", filename);

    // load mtx A data to the csr format
    gettimeofday(&t1, NULL);
    mmio_allinone(&matrixA_d->m, &matrixA_d->n, &matrixA_d->nnz, &matrixA_d->isSymmetric, &matrixA_d->rowpointer, &matrixA_d->columnidx, &matrixA_d->value, filename);
    gettimeofday(&t2, NULL);
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", matrixA_d->m, matrixA_d->n, matrixA_d->nnz, time_loadmat/1000.0);

    for (int i = 0; i < matrixA_d->nnz; i++)
	    matrixA_d->value[i] = i % 10;


    matrixA_d->numtile =0;
    matrixA_d->tilem = matrixA_d->m%BLOCK_SIZE==0 ? matrixA_d->m/BLOCK_SIZE : (matrixA_d->m/BLOCK_SIZE)+1 ;
    matrixA_d->tilen = matrixA_d->n%BLOCK_SIZE==0 ? matrixA_d->n/BLOCK_SIZE : (matrixA_d->n/BLOCK_SIZE)+1 ;
 //MAT_PTR_TYPE *tile_ptr_A_1;    //block rowpointer of A
    matrixA_d->tile_ptr=(MAT_PTR_TYPE *)malloc((matrixA_d->tilem+1)*sizeof(MAT_PTR_TYPE));
    memset(matrixA_d->tile_ptr, 0, (matrixA_d->tilem+1)*sizeof(MAT_PTR_TYPE));

    int *new_coo_rowidx_1;
    int *new_coo_colidx_1;
    MAT_VAL_TYPE *new_coo_value_1;
   
    int *new_coocount_1;

    d_format_transform(matrixA_d, 
                &new_coo_value_1, &new_coo_colidx_1, &new_coo_rowidx_1, &new_coocount_1);
                
    Beidou_Tile_Matrix *matrixA = (Beidou_Tile_Matrix *)malloc(sizeof(Beidou_Tile_Matrix));
        mmio_allinone(&matrixA->m, &matrixA->n, &matrixA->nnz, &matrixA->isSymmetric, &matrixA->rowpointer, &matrixA->columnidx, &matrixA->value, filename);
            matrixA->numtile =0;
    matrixA->tilem = matrixA->m%BLOCK_SIZE==0 ? matrixA->m/BLOCK_SIZE : (matrixA->m/BLOCK_SIZE)+1 ;
    matrixA->tilen = matrixA->n%BLOCK_SIZE==0 ? matrixA->n/BLOCK_SIZE : (matrixA->n/BLOCK_SIZE)+1 ;
 //MAT_PTR_TYPE *tile_ptr_A_1;    //block rowpointer of A
    matrixA->tile_ptr=(MAT_PTR_TYPE *)malloc((matrixA->tilem+1)*sizeof(MAT_PTR_TYPE));
    memset(matrixA->tile_ptr, 0, (matrixA->tilem+1)*sizeof(MAT_PTR_TYPE));
/*int *new_coo_rowidx;
    int *new_coo_colidx;
    MAT_VAL_TYPE *new_coo_value;
    int *new_coocount;
    format_transform(matrixA, 
                  &new_coo_value, &new_coo_colidx, &new_coo_rowidx, &new_coocount);
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA_d->tile_columnidx[i]!=matrixA->tile_columnidx[i]) 
        {
             printf("step2-error-colidx! i=%d   %d!=%d\n",i,matrixA->tile_columnidx[i],matrixA_d->tile_columnidx[i]);
            // break;
        }
        if(matrixA_d->tile_nnz[i]!=matrixA->tile_nnz[i]) 
        {
             printf("step2-error-nnz! i=%d   %d!=%d\n",i,matrixA->tile_nnz[i],matrixA_d->tile_nnz[i]);
             break;
        }
        for(int j=0;j<BLOCK_SIZE;j++)
        {
            if(matrixA->csr_ptr[i*BLOCK_SIZE+j]!=matrixA_d->csr_ptr_1[i*BLOCK_SIZE+j])
            {
               // printf("step2-error-ptr! i=%d   j=%d   %d!=%d\n",i,j,csr_ptr[i*BLOCK_SIZE+j],matrixA_d->csr_ptr_1[i*BLOCK_SIZE+j]);
                break;
            }
        }
        //printf("\n");
    }   */

 //balance
    int rowblkblock_1 = 0;
    int tilecnt_ave_1 =  (double) matrixA_d->tile_ptr[matrixA_d->tilem] / (double) matrixA_d->tilem;
    int *bal_num_gro = (int *) malloc(sizeof(int) * (matrixA_d->tilem +1));
    memset(bal_num_gro, 0, sizeof(int) * (matrixA_d->tilem +1)); 

  
    for (int blki = 0; blki < matrixA_d->tilem; blki++) 
    {
        int balancenumblk = matrixA_d->tile_ptr[blki + 1] - matrixA_d->tile_ptr[blki];
//        printf("balancenumblk = %i\n", balancenumblk); 
        if (balancenumblk <= tilecnt_ave_1)
        {
            bal_num_gro[blki]=rowblkblock_1;
            rowblkblock_1++;  
        }
        else 
        { 
            bal_num_gro[blki]=rowblkblock_1;
            rowblkblock_1 += ceil((double) balancenumblk / (double) tilecnt_ave_1);
        }
    }
    bal_num_gro[matrixA_d->tilem]=rowblkblock_1;
    printf("ave blk num = %4.2f, %i, %i\n", (double) matrixA_d->tile_ptr[matrixA_d->tilem] / (double) matrixA_d->tilem, matrixA_d->tilem, rowblkblock_1);


    matrixA_d->flag_bal_tile_rowidx = (unsigned int *) malloc(sizeof(unsigned int) * rowblkblock_1);
    memset(matrixA_d->flag_bal_tile_rowidx, 0, sizeof(unsigned int) * rowblkblock_1);
    unsigned int *d_flag_bal_tile_rowidx;
    cudaMalloc((void **)&d_flag_bal_tile_rowidx, sizeof(unsigned int) * rowblkblock_1);
    cudaMemcpy(d_flag_bal_tile_rowidx, matrixA_d->flag_bal_tile_rowidx, sizeof(unsigned int) * rowblkblock_1, cudaMemcpyHostToDevice);


    matrixA_d->tile_bal_rowidx_colstart = (int *) malloc(sizeof(int) * rowblkblock_1);
    memset(matrixA_d->tile_bal_rowidx_colstart, 0, sizeof(int) * rowblkblock_1);
    int *d_tile_bal_rowidx_colstart;
    cudaMalloc((void **)&d_tile_bal_rowidx_colstart, sizeof( int) * rowblkblock_1);
    cudaMemcpy(d_tile_bal_rowidx_colstart, matrixA_d->tile_bal_rowidx_colstart, sizeof( int) * rowblkblock_1, cudaMemcpyHostToDevice);

    matrixA_d->tile_bal_rowidx_colstop = (int *) malloc(sizeof(int) * rowblkblock_1);
    memset(matrixA_d->tile_bal_rowidx_colstop, 0, sizeof(int) * rowblkblock_1);
    int *d_tile_bal_rowidx_colstop;
    cudaMalloc((void **)&d_tile_bal_rowidx_colstop, sizeof( int) * rowblkblock_1);
    cudaMemcpy(d_tile_bal_rowidx_colstop, matrixA_d->tile_bal_rowidx_colstop, sizeof( int) * rowblkblock_1, cudaMemcpyHostToDevice);

    int *group_ptr_1 = (int *) malloc(sizeof(int) * (rowblkblock_1 +1));
    memset(group_ptr_1, 0, sizeof(int) * (rowblkblock_1 +1));


    int *d_bal_num_gro;
    cudaMalloc((void **)&d_bal_num_gro, sizeof(int) * (matrixA_d->tilem +1));
    cudaMemcpy(d_bal_num_gro, bal_num_gro, sizeof(int) * (matrixA_d->tilem +1), cudaMemcpyHostToDevice);

    MAT_PTR_TYPE *d_tile_ptr_A;
    cudaMalloc((void **)&d_tile_ptr_A, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1) );
    cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);

    int n_tile=32;
    double time_cuda_bal=0;
    int num_threads=0;
    int num_blocks=0;
    int num_tile_row=64;
    //int num_blocks=0;
    gettimeofday(&t1, NULL);
    
    for(int blki=0;blki<matrixA_d->tilem;blki+=n_tile)
    {   
        int start=blki;
        int end= blki+num_tile_row<matrixA_d->tilem ? end= blki+num_tile_row : end=matrixA_d->tilem;
         
        num_threads= end-start;
        num_blocks=num_threads/64+1;
   
        cuda_bal_step1<<< num_blocks, 64 >>>( matrixA_d->tilem, matrixA_d->tilen, d_tile_ptr_A,start,end,num_threads,
        d_flag_bal_tile_rowidx,d_tile_bal_rowidx_colstart,d_tile_bal_rowidx_colstop,d_bal_num_gro,tilecnt_ave_1);

        cudaDeviceSynchronize();
        
     //   int length=i+n_tile>matrixA_d->tilem ? length=matrixA_d->tilem-i : length=n_tile ;
    }
    gettimeofday(&t2, NULL);
    time_cuda_bal = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
//printf("\n\n\n");
    printf("cuda-balance-step1 runtime    = %4.5f ms\n", time_cuda_bal);
  //   cudaMemcpy(matrixA_d->tile_ptr+i, d_tile_ptr_A+i, sizeof(MAT_PTR_TYPE) *(length+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->tile_bal_rowidx_colstart, d_tile_bal_rowidx_colstart, sizeof( int) * rowblkblock_1, cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->tile_bal_rowidx_colstop, d_tile_bal_rowidx_colstop, sizeof( int) * rowblkblock_1, cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->flag_bal_tile_rowidx, d_flag_bal_tile_rowidx, sizeof(unsigned int) * rowblkblock_1, cudaMemcpyDeviceToHost);


    for (int i = 0; i < rowblkblock_1; i ++)
    {
        int tile_start = matrixA_d->tile_bal_rowidx_colstart[i];
        int tile_stop = matrixA_d->tile_bal_rowidx_colstop[i];
        group_ptr_1[i] = matrixA_d->blknnz[tile_stop] - matrixA_d->blknnz[tile_start];
    }
    
    exclusive_scan(group_ptr_1,rowblkblock_1 +1);

    int *d_group_ptr;
    cudaMalloc((void **)&d_group_ptr, sizeof( int) * rowblkblock_1);
    cudaMemcpy(d_group_ptr, group_ptr_1, sizeof( int) * rowblkblock_1, cudaMemcpyHostToDevice);

    int *flag_tilerow_start_1 = (int *)malloc((nthreads + 1) * sizeof(int));
    memset(flag_tilerow_start_1, 0, (nthreads) * sizeof(int));
    int *d_flag_tilerow_start;
    cudaMalloc((void **)&d_flag_tilerow_start, (nthreads + 1) * sizeof(int));
    cudaMemcpy(d_flag_tilerow_start, flag_tilerow_start_1, (nthreads + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int *flag_tilerow_stop_1 = (int *)malloc((nthreads) * sizeof(int));
    memset(flag_tilerow_stop_1, 0, (nthreads) * sizeof(int));

    //int *csrSplitter_normal = (int *)malloc((nthreads+1) * sizeof(int));
    int stridennz_1 = ceil((double)matrixA_d->nnz/(double)nthreads);

    
    double time_cuda_bal_1=0;
    gettimeofday(&t1, NULL);
    cuda_bal_step2<<< nthreads/64+1, 64 >>>( stridennz_1, matrixA_d->nnz,nthreads,rowblkblock_1,d_flag_tilerow_start,d_group_ptr);
    gettimeofday(&t2, NULL);
    cudaDeviceSynchronize();
    time_cuda_bal_1 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cuda-balance-step2 runtime    = %4.5f ms\n", time_cuda_bal_1);
    cudaMemcpy(flag_tilerow_start_1, d_flag_tilerow_start, (nthreads + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    for (int tid =0; tid < nthreads -1; tid ++)
    {
        flag_tilerow_stop_1[tid] = flag_tilerow_start_1[tid + 1];
    }
    flag_tilerow_stop_1[nthreads -1] = rowblkblock_1;




    for(int tid =0; tid < nthreads; tid ++)
    {
        printf("cuda-thread %i start = %i, stop = %i\n", tid, flag_tilerow_start_1[tid],flag_tilerow_stop_1[tid]);
       // printf("thread %i start = %i, stop = %i\n", tid, flag_tilerow_start_1[tid],flag_tilerow_stop_1[tid]);
    }    

    //extract coo data to a new matrix
    
    int nnz_1 = matrixA_d->coocount;
    printf("cuda-the number of coo data  = %i\n",nnz_1);

    double ratio_1 = (double)nnz_1/(double)matrixA_d->nnz;

    printf("cuda-the ratio of coo data  = %f\n",ratio_1);

    int *new_nnz_count_1 = (int *)malloc((matrixA_d->m+1) * sizeof(int));
    memset(new_nnz_count_1, 0,(matrixA_d->m+1) * sizeof(int));

    int *d_new_nnz_count;
    cudaMalloc((void **)&d_new_nnz_count, (matrixA_d->m+1) * sizeof(int));
    cudaMemcpy(d_new_nnz_count, new_nnz_count_1, (matrixA_d->m+1) * sizeof(int), cudaMemcpyHostToDevice);

    int *d_new_coo_rowidx;
    cudaMalloc((void **)&d_new_coo_rowidx, (matrixA_d->coocount) *sizeof(int));
    cudaMemcpy(d_new_coo_rowidx, new_coo_rowidx_1, (matrixA_d->coocount) *sizeof(int), cudaMemcpyHostToDevice);

    int num_nnz_1=4096;
    gettimeofday(&t1, NULL);

    for (int i=0;i<nnz_1;i+=num_nnz_1)//
    {
        //cudaMemset(d_col_flag,0,tilenA*num_tile_row*16 * sizeof(unsigned char));
        int start=i;
        int end= i+num_nnz_1>nnz_1? end=nnz_1-1 : end=i+num_nnz_1-1;
        int num_threads= i+num_nnz_1>nnz_1? num_threads=nnz_1-i : num_threads=num_nnz_1;
        //int end= blki+num_tile_row<tilemA ? end= blki+num_tile_row : end=tilemA;
        num_blocks=(end-start)/32+1;
        //printf("end=%d  start=%d  num_blocks=%d  blki=%d\n",end,start,num_blocks,blki);
        cuda_coo_rowptrnum_kernel<<<num_blocks, 32 >>>(nnz_1,matrixA_d->m, matrixA_d->n,d_new_nnz_count,start,end,d_new_coo_rowidx,num_threads);
       cudaDeviceSynchronize();
        
    }
    gettimeofday(&t2, NULL);
    cudaDeviceSynchronize();
    double cuda_time_coo  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cuda_transform_coo_1 runtime    = %4.5f ms\n", cuda_time_coo);
    cudaMemcpy(new_nnz_count_1, d_new_nnz_count, (matrixA_d->m+1) * sizeof(int), cudaMemcpyDeviceToHost);

    int new_row_1 =0;

    //int *coo_new_rowidx_1 = (int *)malloc((matrixA_d->m ) * sizeof(int));
    matrixA_d->coo_new_rowidx = (int *)malloc((matrixA_d->m ) * sizeof(int));
    memset(matrixA_d->coo_new_rowidx, 0, matrixA_d->m * sizeof(int));

    for (int i=0; i < matrixA_d->m ; i ++)
    {
        if (new_nnz_count_1[i] !=0){
            matrixA_d->coo_new_rowidx[new_row_1++] = i;
        }
    }

    //int *coo_new_matrix_ptr_1 = (int*)malloc(sizeof(int)*(new_row_1+1));
    matrixA_d->coo_new_matrix_ptr = (int*)malloc(sizeof(int)*(new_row_1+1));
    memset(matrixA_d->coo_new_matrix_ptr, 0, (new_row_1 + 1) * sizeof(int));
    int cnt_1 =0;

    for (int i=0; i < matrixA_d->m ; i ++)
    {
        if (new_nnz_count_1[i] !=0){
            matrixA_d->coo_new_matrix_ptr[cnt_1++] = new_nnz_count_1[i] ;
        }
    }
    exclusive_scan(matrixA_d->coo_new_matrix_ptr,new_row_1+1);
    exclusive_scan(new_nnz_count_1, matrixA_d->m +1);
   // exclusive_scan(new_nnz_count_1, rowA +1);

    matrixA_d->coo_new_matrix_colidx  = (int *)malloc(nnz_1 * sizeof(int));
    memset(matrixA_d->coo_new_matrix_colidx, 0,  nnz_1 * sizeof(int));
    int *d_coo_new_colidx;
    cudaMalloc((void **)&d_coo_new_colidx, nnz_1 * sizeof(int));
    cudaMemcpy(d_coo_new_colidx, matrixA_d->coo_new_matrix_colidx, nnz_1 * sizeof(int), cudaMemcpyHostToDevice);
//coo_new_matrix_ptr
    //MAT_VAL_TYPE *coo_new_value_1 = (MAT_VAL_TYPE *)malloc(nnz_1 * sizeof(MAT_VAL_TYPE));
    matrixA_d->coo_new_matrix_value = (MAT_VAL_TYPE *)malloc(nnz_1 * sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->coo_new_matrix_value, 0,  nnz_1 * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_coo_new_value;
    cudaMalloc((void **)&d_coo_new_value, nnz_1 * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_coo_new_value, matrixA_d->coo_new_matrix_value, nnz_1 * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    int *new_num_1  = (int *)malloc( matrixA_d->m* sizeof(int));//new_row_1
    memset(new_num_1, 0,  matrixA_d->m * sizeof(int));
    int *d_new_num;
    cudaMalloc((void **)&d_new_num, matrixA_d->m* sizeof(int));
    cudaMemcpy(d_new_num, new_num_1, matrixA_d->m* sizeof(int), cudaMemcpyHostToDevice);
   // cudaMemcpy(d_new_coo_colidx, new_coo_colidx_1, (matrixA_d->coocount) *sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_new_coo_value, new_coo_value_1, (matrixA_d->coocount) *sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
  
    int *d_new_coo_colidx;
    cudaMalloc((void **)&d_new_coo_colidx, (nnz_1) *sizeof(int));
    cudaMemcpy(d_new_coo_colidx, new_coo_colidx_1, (nnz_1) *sizeof(int), cudaMemcpyHostToDevice);

    MAT_VAL_TYPE *d_new_coo_value;
    cudaMalloc((void **)&d_new_coo_value, (nnz_1) *sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_new_coo_value, new_coo_value_1, (nnz_1) *sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_new_coo_value, new_coo_value_1, (matrixA_d->coocount) *sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_nnz_count, new_nnz_count_1, (matrixA_d->m+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_coo_rowidx, new_coo_rowidx_1, (matrixA_d->coocount) *sizeof(int), cudaMemcpyHostToDevice);
    gettimeofday(&t1, NULL);
   // int x=0;
    for (int i=0;i<nnz_1;i+=num_nnz_1)//
    {
        //cudaMemset(d_col_flag,0,tilenA*num_tile_row*16 * sizeof(unsigned char));
        int start=i;
        int end= i+num_nnz_1>nnz_1? end=nnz_1-1 : end=i+num_nnz_1-1;
        int num_threads= i+num_nnz_1>nnz_1? num_threads=nnz_1-i : num_threads=num_nnz_1;
        //int end= blki+num_tile_row<tilemA ? end= blki+num_tile_row : end=tilemA;
        num_blocks=(end-start)/32+1;
       // printf("end=%d  start=%d \n",end,start);
        cuda_coo_kernel<<<num_blocks, 32 >>>(nnz_1,matrixA_d->m, matrixA_d->n,d_new_nnz_count,start,end,d_new_coo_rowidx,num_threads,
                 d_coo_new_colidx,d_coo_new_value,d_new_num,d_new_coo_value,d_new_coo_colidx);
       cudaDeviceSynchronize();
        
    }
    gettimeofday(&t2, NULL);
    cudaDeviceSynchronize();
    double cuda_time_coo_1  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cuda_transform_coo_2 runtime    = %4.5f ms\n", cuda_time_coo_1);
    cudaMemcpy(matrixA_d->coo_new_matrix_colidx, d_coo_new_colidx, nnz_1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->coo_new_matrix_value, d_coo_new_value, nnz_1 * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_num_1, d_new_num, matrixA_d->m* sizeof(int), cudaMemcpyDeviceToHost);
//printf("new_coo_rowidx[nnz]=%d\n",new_coo_rowidx[matrixA_d->coocount-1]);
 //printf("hhhh\n");
//sort
    gettimeofday(&t1, NULL);
 //  #pragma omp parallel for
   // for (int i =0 ; i < new_row_1; i ++)
    //{
      //  int nnz_offset = coo_new_matrix_ptr_1[i];
        //int length = coo_new_matrix_ptr_1[i+1] - coo_new_matrix_ptr_1[i];

      //  quick_sort_key_val_pair(coo_new_colidx_1 + nnz_offset, coo_new_value_1 + nnz_offset, length);
   // }
    //thrust::sort_by_key(new_coo_colidx_1 , new_coo_colidx_1 +nnz_1, new_coo_value_1);
   // sort_by_row_and_column(new_coo_rowidx_1,new_coo_colidx_1,new_coo_value_1,0,nnz_1,0,nnz_1);
    for (int i =0 ; i < new_row_1; i ++)
    {
        int nnz_offset = matrixA_d->coo_new_matrix_ptr[i];
        int length = matrixA_d->coo_new_matrix_ptr[i+1] - matrixA_d->coo_new_matrix_ptr[i];

        //quick_sort_key_val_pair(matrixA_d->coo_new_matrix_colidx + nnz_offset, matrixA_d->coo_new_matrix_value + nnz_offset, length);
        thrust::sort_by_key(matrixA_d->coo_new_matrix_colidx + nnz_offset, matrixA_d->coo_new_matrix_colidx + nnz_offset + length, matrixA_d->coo_new_matrix_value + nnz_offset);
    }

    gettimeofday(&t2, NULL);
    double time_sort_1  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cuda-sort use   %4.5f ms\n", time_sort_1);


printf("--------------------------------!!-cuda-end-!!------------------------------------\n");

    MAT_VAL_TYPE *x = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * matrixA_d->n);
	for (int i = 0; i < matrixA_d->n; i++)
	{
		x[i] = i % 10;
	}
MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * matrixA_d->m);
    memset(y, 0, sizeof(MAT_VAL_TYPE) * matrixA_d->m);

    gettimeofday(&t1, NULL);
    for (int repeat =0; repeat < BENCH_REPEAT; repeat ++)
    {
            memset(y, 0, sizeof(MAT_VAL_TYPE) * matrixA_d->m);

        tilespmv(matrixA_d,  x, y,new_row_1);

    }
    gettimeofday(&t2, NULL);
    double time_tile  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("spmv runtime using tile    = %4.5f ms\n", time_tile/BENCH_REPEAT);
    
    MAT_VAL_TYPE *y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * matrixA_d->m);
    memset(y_golden, 0, sizeof(MAT_VAL_TYPE) * matrixA_d->m);
    for (int i = 0; i < matrixA_d->n; i++)
	{
		MAT_VAL_TYPE sum = 0;
		for (int j = matrixA_d->rowpointer[i]; j < matrixA_d->rowpointer[i+1]; j++)
		{
			sum += matrixA_d->value[j] * x[matrixA_d->columnidx[j]];
		}
		y_golden[i] = sum;
	}

printf("--------------------------------!!-bal-!!------------------------------------\n");
    //int *csr_ptr_1 = (int *)malloc(((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int));
    matrixA_d->csr_ptr = (unsigned char *)malloc(((matrixA_d->numtile+1) * BLOCK_SIZE) * sizeof(unsigned char));
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        matrixA_d->csr_ptr[i]=matrixA_d->csr_ptr_1[i];
    }
	double *Ysum_1 = (double *)malloc(sizeof(double) * nthreads);
	memset (Ysum_1, 0, sizeof(double) * nthreads);
    double *Ypartialsum_1 = (double *)malloc(sizeof(double) * nthreads);
	memset (Ypartialsum_1, 0, sizeof(double) * nthreads);
    MAT_VAL_TYPE *y_bal_1 = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * matrixA_d->m);
    memset(y_bal_1, 0, sizeof(MAT_VAL_TYPE) * matrixA_d->m);

    matrixA_d->csrSplitter_yid = (int *)malloc((nthreads+1) * sizeof(int));
    matrixA_d->Yid = (int *)malloc(sizeof(int) * nthreads);
    matrixA_d->Start1 = (int *)malloc(sizeof(int) * nthreads);
	matrixA_d->End1 = (int *)malloc(sizeof(int) * nthreads);
	matrixA_d->label = (int *)malloc(sizeof(int) * nthreads);
    matrixA_d->Start2 = (int *)malloc(sizeof(int) * nthreads);
	matrixA_d->End2 = (int *)malloc(sizeof(int) * nthreads);
    /*tilespmv_balance(matrixA_d, rowblkblock_1,  csr_ptr_1, hyb_coocount_1, nnz_temp, tile_count_temp,csr_offset_1, csrptr_offset_1, 
    coo_offset_1, ell_offset_1, hyb_offset_1, dns_offset_1, dnsrow_offset_1, dnscol_offset_1, x, y_bal_1,
    flag_tilerow_start_1, flag_tilerow_stop_1);*/
    spmvLBLT(new_row_1,nthreads, matrixA_d->m, matrixA_d->n, matrixA_d->nnz, matrixA_d->coo_new_rowidx,matrixA_d->coo_new_matrix_ptr,
            matrixA_d->coo_new_matrix_colidx,matrixA_d->coo_new_matrix_value,matrixA_d->csrSplitter_yid,matrixA_d->Yid,
            matrixA_d->Start1,matrixA_d->End1,matrixA_d->label,matrixA_d->Start2,matrixA_d->End2);
    //printf("hhhhhhhhhhh\n");
    tilespmv_balance(matrixA_d, rowblkblock_1, x, y_bal_1,
                        flag_tilerow_start_1, flag_tilerow_stop_1, Ysum_1, Ypartialsum_1);
    int balcnt_1 =0;
//printf("hhhhhhhhhhh\n");
    for (int i=0; i < matrixA_d->m; i ++)//
    {
        if (y_golden[i] != y_bal_1[i])
        {
          // printf("y[%i] = %f, y_bal_1[%i] = %f\n", i, y[i], i, y_bal_1[i]);
            balcnt_1 ++;
        }
    }

    printf("cuda-balance result errcnt = %i\n",balcnt_1);
    //printf("omp_get_max_threads()=%d\n",omp_get_max_threads());
printf("--------------------------------!!-bal end-!!------------------------------------\n");
printf("--------------------------------!!-cuda spmv-!!------------------------------------\n");
	char *flag=(char *)malloc(matrixA_d->tilen*sizeof(char));
	int nnzbl=0;

	for (int i=0;i<matrixA_d->tilem;i++)
	{
		memset(flag,0,matrixA_d->tilen*sizeof(char));
		int start= i*BLOCK_SIZE;
		int end = i==matrixA_d->tilem-1 ?  matrixA_d->m : (i+1)*BLOCK_SIZE ;
		for (int j=matrixA_d->rowpointer[start];j<matrixA_d->rowpointer[end];j++)
		{
			int jc=matrixA_d->columnidx[j]/BLOCK_SIZE;
			if (flag[jc]==0)
			{
				flag[jc]=1;
				nnzbl++;
			}
		} 
	}
	int colid=0;
    int ptrA_length=0;
    for (int i=0;i<matrixA_d->tilem;i++)
	{
        memset(flag,0,matrixA_d->tilen*sizeof(char));
        int start= i*BLOCK_SIZE;
        int end = i==matrixA_d->tilem-1 ?  matrixA_d->m : (i+1)*BLOCK_SIZE ;
		for (int j=matrixA_d->rowpointer[start];j<matrixA_d->rowpointer[end];j++)
        {
            int jc=matrixA_d->columnidx[j]/BLOCK_SIZE;
            if (flag[jc]==0)
            {
                flag[jc]=1;
              //  rowblock_ptr[i+1]++;
             //   columnid[colid]=jc;
                colid++;
                ptrA_length+=(end-start);
            }
	    } 
	}
    char *d_flag; 
	cudaMalloc((void **)&d_flag, matrixA_d->tilen*sizeof(char));
	cudaMemcpy(d_flag, flag, matrixA_d->tilen*sizeof(char), cudaMemcpyHostToDevice);

    MAT_VAL_TYPE *y_d = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * matrixA_d->m);
    memset(y_d, 0, sizeof(MAT_VAL_TYPE) * matrixA_d->m);

    MAT_VAL_TYPE *d_x;
    MAT_VAL_TYPE *d_y;

    cudaMalloc((void **)&d_x, matrixA_d->n * sizeof(MAT_VAL_TYPE)); 
    cudaMalloc((void **)&d_y, matrixA_d->m * sizeof(MAT_VAL_TYPE));

    cudaMemcpy(d_x, x, matrixA_d->n * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_d, matrixA_d->m * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);



  //  MAT_PTR_TYPE *d_tile_ptr_A;
  //  cudaMalloc((void **)&d_tile_ptr_A, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1) );
    cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);

    int *d_tile_columnidx;
    cudaMalloc((void **)&d_tile_columnidx, (matrixA_d->numtile + 1) * sizeof(int) );
    cudaMemcpy(d_tile_columnidx, matrixA_d->tile_columnidx, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);

    int *d_tile_nnz;
    cudaMalloc((void **)&d_tile_nnz, (matrixA_d->numtile + 1) * sizeof(int) );
    cudaMemcpy(d_tile_nnz, matrixA_d->tile_nnz, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);

    char *d_Format;
    cudaMalloc((void **)&d_Format, matrixA_d->numtile* sizeof(char) );
    cudaMemcpy(d_Format, matrixA_d->Format, matrixA_d->numtile* sizeof(char), cudaMemcpyHostToDevice);

    int *d_blknnz;
    cudaMalloc((void **)&d_blknnz, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_blknnz, matrixA_d->blknnz, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    int *d_dns_offset;
    cudaMalloc((void **)&d_dns_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_dns_offset, matrixA_d->dns_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
printf("matrixA_d->dns_offset=%d  %d\n",matrixA_d->dns_offset[matrixA_d->numtile],matrixA_d->dns_offset[matrixA_d->numtile-1]);


    int *d_denserowptr;
    cudaMalloc((void **)&d_denserowptr, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_denserowptr, matrixA_d->denserowptr, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
    
    int *d_dnsrow_offset;
    cudaMalloc((void **)&d_dnsrow_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_dnsrow_offset, matrixA_d->dnsrow_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
    
    int *d_densecolptr;
    cudaMalloc((void **)&d_densecolptr, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_densecolptr, matrixA_d->densecolptr, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
    
    int *d_dnscol_offset;
    cudaMalloc((void **)&d_dnscol_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_dnscol_offset, matrixA_d->dnscol_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    int *d_csr_offset;
    cudaMalloc((void **)&d_csr_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_csr_offset, matrixA_d->csr_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    int *d_csrptr_offset;
    cudaMalloc((void **)&d_csrptr_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_csrptr_offset, matrixA_d->csrptr_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    int *d_ell_offset;
    cudaMalloc((void **)&d_ell_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_ell_offset, matrixA_d->ell_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    int *d_coo_offset;
    cudaMalloc((void **)&d_coo_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_coo_offset, matrixA_d->coo_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    char *d_blkwidth;
    cudaMalloc((void **)&d_blkwidth, (matrixA_d->numtile + 1)* sizeof(char) );
    cudaMemcpy(d_blkwidth, matrixA_d->blkwidth, (matrixA_d->numtile + 1)* sizeof(char), cudaMemcpyHostToDevice);

    int *d_hyb_coocount;
    cudaMalloc((void **)&d_hyb_coocount, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_hyb_coocount, matrixA_d->hyb_coocount, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    int *d_hyb_offset;
    cudaMalloc((void **)&d_hyb_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_hyb_offset, matrixA_d->hyb_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

//CSR
    MAT_VAL_TYPE *d_Tile_csr_Val;
    cudaMalloc((void **)&d_Tile_csr_Val, (matrixA_d->csrsize)*sizeof(MAT_VAL_TYPE) );
    cudaMemcpy(d_Tile_csr_Val, matrixA_d->Tile_csr_Val, (matrixA_d->csrsize)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    unsigned char  *d_Tile_csr_Col;
    cudaMalloc((void **)&d_Tile_csr_Col,(matrixA_d->csrsize)*sizeof(unsigned char) );
    cudaMemcpy(d_Tile_csr_Col, matrixA_d->Tile_csr_Col, (matrixA_d->csrsize)*sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsigned char *d_Tile_csr_Ptr;
    cudaMalloc((void **)&d_Tile_csr_Ptr,(matrixA_d->csrtilecount * BLOCK_SIZE)*sizeof(unsigned char) );
    cudaMemcpy(d_Tile_csr_Ptr, matrixA_d->Tile_csr_Ptr,(matrixA_d->csrtilecount * BLOCK_SIZE)*sizeof(unsigned char), cudaMemcpyHostToDevice);

//ELL
    MAT_VAL_TYPE *d_Tile_ell_Val;
    cudaMalloc((void **)&d_Tile_ell_Val, (matrixA_d->ellsize)*sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_ell_Val, matrixA_d->Tile_ell_Val, (matrixA_d->ellsize)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    unsigned char *d_Tile_ell_colIdx;
    cudaMalloc((void **)&d_Tile_ell_colIdx, (matrixA_d->ellsize)*sizeof(unsigned char));
    cudaMemcpy(d_Tile_ell_colIdx, matrixA_d->Tile_ell_colIdx, (matrixA_d->ellsize)*sizeof(unsigned char), cudaMemcpyHostToDevice);

//HYB
    MAT_VAL_TYPE *d_Tile_hyb_Val;
    cudaMalloc((void **)&d_Tile_hyb_Val, (matrixA_d->hybellsize+matrixA_d->hybcoosize)*sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_hyb_Val, matrixA_d->Tile_hyb_Val, (matrixA_d->hybellsize+matrixA_d->hybcoosize)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    unsigned char *d_Tile_hyb_ellcolIdx;
    cudaMalloc((void **)&d_Tile_hyb_ellcolIdx, (matrixA_d->hybellsize+matrixA_d->hybcoosize)*sizeof(unsigned char));
    cudaMemcpy(d_Tile_hyb_ellcolIdx, matrixA_d->Tile_hyb_ellcolIdx, (matrixA_d->hybellsize+matrixA_d->hybcoosize)*sizeof(unsigned char), cudaMemcpyHostToDevice);


    unsigned char *d_Tile_hyb_coorowIdx;
    cudaMalloc((void **)&d_Tile_hyb_coorowIdx, (matrixA_d->hybcoosize)*sizeof(unsigned char));
    cudaMemcpy(d_Tile_hyb_coorowIdx, matrixA_d->Tile_hyb_coorowIdx, (matrixA_d->hybcoosize)*sizeof(unsigned char), cudaMemcpyHostToDevice);

//dense
    MAT_VAL_TYPE *d_Tile_dns_Val;
    cudaMalloc((void **)&d_Tile_dns_Val, (matrixA_d->dense_size)*sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_dns_Val, matrixA_d->Tile_dns_Val, (matrixA_d->dense_size)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

//dense row
    MAT_VAL_TYPE *d_Tile_dnsrow_Val;
    cudaMalloc((void **)&d_Tile_dnsrow_Val, (matrixA_d->denserow_size) * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_dnsrow_Val, matrixA_d->Tile_dnsrow_Val, (matrixA_d->denserow_size) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    char *d_Tile_dnsrow_idx ;
    cudaMalloc((void **)&d_Tile_dnsrow_idx, matrixA_d->denserowptr[matrixA_d->numtile] * sizeof(char));
    cudaMemcpy(d_Tile_dnsrow_idx, matrixA_d->Tile_dnsrow_idx, matrixA_d->denserowptr[matrixA_d->numtile] * sizeof(char), cudaMemcpyHostToDevice);

//dense col
    MAT_VAL_TYPE *d_Tile_dnscol_Val;
    cudaMalloc((void **)&d_Tile_dnscol_Val, (matrixA_d->densecol_size) * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_dnscol_Val, matrixA_d->Tile_dnscol_Val, (matrixA_d->densecol_size) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    char *d_Tile_dnscol_idx;
    cudaMalloc((void **)&d_Tile_dnscol_idx, matrixA_d->densecolptr[matrixA_d->numtile] * sizeof(char));
    cudaMemcpy(d_Tile_dnscol_idx, matrixA_d->Tile_dnscol_idx, matrixA_d->densecolptr[matrixA_d->numtile] * sizeof(char), cudaMemcpyHostToDevice);

//COO
 //   MAT_VAL_TYPE *d_coo_new_value;
 //   cudaMalloc((void **)&d_coo_new_value, nnz_1 * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_coo_new_value, matrixA_d->coo_new_matrix_value, nnz_1 * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    int *d_coo_new_rowidx;
    cudaMalloc((void **)&d_coo_new_rowidx, (matrixA_d->m ) * sizeof(int));
    cudaMemcpy(d_coo_new_rowidx, matrixA_d->coo_new_rowidx, (matrixA_d->m ) * sizeof(int), cudaMemcpyHostToDevice);

    int *d_coo_new_matrix_ptr;
    cudaMalloc((void **)&d_coo_new_matrix_ptr, sizeof(int)*(new_row_1+1));
    cudaMemcpy(d_coo_new_matrix_ptr, matrixA_d->coo_new_matrix_ptr, sizeof(int)*(new_row_1+1), cudaMemcpyHostToDevice);

    int *d_coo_new_matrix_colidx;
    cudaMalloc((void **)&d_coo_new_matrix_colidx, nnz_1 * sizeof(int));
    cudaMemcpy(d_coo_new_matrix_colidx, matrixA_d->coo_new_matrix_colidx, nnz_1 * sizeof(int), cudaMemcpyHostToDevice);

    unsigned char *d_blknnznnz;
    cudaMalloc((void **)&d_blknnznnz, (matrixA_d->numtile + 1)* sizeof(unsigned char));
    cudaMemcpy(d_blknnznnz, matrixA_d->blknnznnz, (matrixA_d->numtile + 1)* sizeof(unsigned char), cudaMemcpyHostToDevice);

    int *d_coodeferoffset;
    int *d_deferbuf_coooff;
    int *d_deferbuf_dxoff;

    cudaMalloc((void **)&d_coodeferoffset, rowblkblock_1 * sizeof(int));
    cudaMemset(d_coodeferoffset, 0, rowblkblock_1 * sizeof(int));

    cudaMalloc((void **)&d_deferbuf_coooff, rowblkblock_1 * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMemset(d_deferbuf_coooff, 0, rowblkblock_1 * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMalloc((void **)&d_deferbuf_dxoff, rowblkblock_1 * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));
    cudaMemset(d_deferbuf_dxoff, 0, rowblkblock_1 * PREFETCH_SMEM_TH * COO_NNZ_TH * sizeof(int));

    //int *d_flag_tilerow_start;
    //cudaMalloc((void **)&d_flag_tilerow_start, (nthreads + 1) * sizeof(int));
    cudaMemcpy(d_flag_tilerow_start, flag_tilerow_start_1, (rowblkblock_1 + 1) * sizeof(int), cudaMemcpyHostToDevice);
    int *d_flag_tilerow_stop;
    cudaMalloc((void **)&d_flag_tilerow_stop, (rowblkblock_1 ) * sizeof(int));
    cudaMemcpy(d_flag_tilerow_stop, flag_tilerow_stop_1, (rowblkblock_1 ) * sizeof(int), cudaMemcpyHostToDevice);

//-------------------
    // analysis
    int rowblkblock = 0;
    //int iiiii = 0;
    for (int blki = 0; blki < matrixA_d->tilem; blki++)
    {
        int balancenumblk = matrixA_d->tile_ptr[blki+1] - matrixA_d->tile_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH) 
            rowblkblock++;
        else 
        {
            rowblkblock += ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH);
            //printf("[%i] blki = %i, balancenumblk = %i, rowblkblock += %i\n", iiiii, blki, balancenumblk, balancenumblk / 32); 
            //iiiii++;
        }

    }
    printf("ave blk num = %4.2f, %i, %i\n", (double)matrixA_d->tile_ptr[matrixA_d->tilem] / (double)matrixA_d->tilem, matrixA_d->tilem, rowblkblock);

    unsigned int * blkcoostylerowidx = (unsigned int *)malloc(sizeof(unsigned int) * rowblkblock);
    memset(blkcoostylerowidx, 0, sizeof(unsigned int) * rowblkblock);
    int * blkcoostylerowidx_colstart = (int *)malloc(sizeof(int) * rowblkblock);
    memset(blkcoostylerowidx_colstart, 0, sizeof(int) * rowblkblock);
    int * blkcoostylerowidx_colstop = (int *)malloc(sizeof(int) * rowblkblock);
    memset(blkcoostylerowidx_colstop, 0, sizeof(int) * rowblkblock);

    int rowblkblockcnt = 0;
    for (int blki = 0; blki < matrixA_d->tilem; blki++)
    {
        int balancenumblk = matrixA_d->tile_ptr[blki+1] - matrixA_d->tile_ptr[blki];
//printf("blki=%d  balancenumblk=%d\n",blki,balancenumblk);
        if (balancenumblk <= PREFETCH_SMEM_TH) 
        {
            blkcoostylerowidx[rowblkblockcnt] = blki;
            rowblkblockcnt++;
        }
        else 
        {
            int numblklocal = ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH);
            int lenblklocal = ceil((double)balancenumblk / (double)numblklocal);
            for (int iii = 0; iii < numblklocal; iii++)
            {
                blkcoostylerowidx[rowblkblockcnt] = blki | 0x80000000; // can generate -0
                blkcoostylerowidx_colstart[rowblkblockcnt] = matrixA_d->tile_ptr[blki] + iii * lenblklocal;
                if (iii == numblklocal - 1)
                    blkcoostylerowidx_colstop[rowblkblockcnt] = matrixA_d->tile_ptr[blki] + balancenumblk;
                else 
                    blkcoostylerowidx_colstop[rowblkblockcnt] = matrixA_d->tile_ptr[blki] + (iii+1) * lenblklocal;

                rowblkblockcnt++;
            }
        }

    }



    unsigned int * d_blkcoostylerowidx;
    int * d_blkcoostylerowidx_colstart;
    int * d_blkcoostylerowidx_colstop;

    cudaMalloc((void **)&d_blkcoostylerowidx, rowblkblock * sizeof(unsigned int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstart, rowblkblock * sizeof(int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstop, rowblkblock * sizeof(int));

    cudaMemcpy(d_blkcoostylerowidx, blkcoostylerowidx, rowblkblock * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstart, blkcoostylerowidx_colstart, rowblkblock * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstop, blkcoostylerowidx_colstop, rowblkblock * sizeof(int), cudaMemcpyHostToDevice);
//-------------------

    //num_threads = WARP_PER_BLOCK * WARP_SIZE;
    //num_blocks = ceil ((double)rowblkblock_1 / (double)(num_threads / WARP_SIZE));

    double time_cuda_spmv6 = 0;
    for (int i = 0; i < 1; i++)//
    {
        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
        int num_blocks = ceil ((double)rowblkblock / (double)(num_threads / WARP_SIZE));
      //  printf("num_threads=%d  num_blocks=%d rowblkblock_1=%d  rowblkblock=%d\n",num_threads,num_blocks,rowblkblock_1,rowblkblock);
        cudaMemset(d_y, 0, matrixA_d->m * sizeof(MAT_VAL_TYPE));
       // rowblkblock_1=10057;
        gettimeofday(&t1, NULL);
        stir_spmv_cuda_kernel_v6<<< num_blocks, num_threads >>>
                (matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,matrixA_d->dense_size,
                d_tile_nnz,  d_flag,  d_tile_ptr_A,  d_tile_columnidx,  d_Format,  d_blknnz, d_blknnznnz,
                d_Tile_csr_Col, d_Tile_csr_Val,  d_Tile_csr_Ptr, 
                d_blkwidth, d_Tile_ell_Val, d_Tile_ell_colIdx, 
                d_Tile_hyb_ellcolIdx, d_Tile_hyb_Val, 
                d_Tile_dns_Val, 
                d_denserowptr,  d_Tile_dnsrow_Val,  d_Tile_dnsrow_idx, 
                d_densecolptr,  d_Tile_dnscol_Val,  d_Tile_dnscol_idx, 
                d_dns_offset, d_dnsrow_offset, d_dnscol_offset, d_csr_offset, d_csrptr_offset, d_ell_offset, d_coo_offset, d_hyb_coocount, d_hyb_offset,
                rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
//d_flag_bal_tile_rowidx, d_flag_tilerow_start, d_flag_tilerow_stop,
                d_coo_new_rowidx, d_coo_new_matrix_colidx, d_coo_new_value,d_coo_new_matrix_ptr,
                d_x,  d_y, 7, d_coodeferoffset, d_deferbuf_coooff, d_deferbuf_dxoff);
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_cuda_spmv6 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        num_threads = WARP_PER_BLOCK * WARP_SIZE;
        num_blocks = ceil ((double)new_row_1 / (double)num_threads);

    cudaMemcpy(y_d, d_y, matrixA_d->m * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);

  //  cudaMemcpy(d_y, y_d, matrixA_d->m * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
cudaMemcpy(d_y, y_d, matrixA_d->m * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
gettimeofday(&t1, NULL);
        spmv_coo<<< num_blocks, num_threads >>>(d_coo_new_matrix_ptr, d_coo_new_matrix_colidx, d_coo_new_value,d_coo_new_rowidx,matrixA_d->m,new_row_1,d_x,  d_y);

       cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
time_cuda_spmv6 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;


//printf("num_threads=%d  num_blocks=%d rowblkblock_1=%d  rowblkblock=%d\n",num_threads,num_blocks,rowblkblock_1,rowblkblock);
    }

  //  time_cuda_spmv6 /= BENCH_REPEAT;
    printf("\n  CUDA SpMV V6 %f ms, %f GFlops\n", 
        time_cuda_spmv6, 2 * (double)matrixA_d->nnz * 1.0e-6 / time_cuda_spmv6);


cudaMemcpy(y_d, d_y, matrixA_d->m * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);




    int cudcnt=0;
    for (int i=0; i < matrixA_d->m; i ++)//

    {
        if (y_golden[i] != y_d[i])
        {
           //printf("y[%i] = %f, y_d[%i] = %f    %f\n", i, y[i], i, y_d[i],y[i]-y_d[i]);
            cudcnt ++;
        }
    }

    printf("cuda-balance result errcnt = %i\n",cudcnt);


    FILE *fouttime = fopen("beidouspmv_res.csv", "a");
	fprintf(fouttime, "%s,%i,%i,%i,%i,%f,%f,%i\n",
			filename, matrixA_d->m, matrixA_d->n, matrixA_d->nnz, matrixA_d->numtile,
			time_cuda_spmv6, 2 * (double)matrixA_d->nnz * 1.0e-6 / time_cuda_spmv6, cudcnt);
fclose(fouttime);
    cudaFree(d_tile_ptr_A);
    cudaFree(d_flag);
    cudaFree(d_tile_columnidx);
    cudaFree(d_tile_nnz);
    cudaFree(d_Format);
    cudaFree(d_blknnz);
    cudaFree(d_blknnznnz);

    cudaFree(d_dns_offset);
    cudaFree(d_denserowptr);
    cudaFree(d_dnsrow_offset);
    cudaFree(d_densecolptr);
    cudaFree(d_dnscol_offset);
    cudaFree(d_csr_offset);
    cudaFree(d_csrptr_offset);
    cudaFree(d_ell_offset);
    cudaFree(d_coo_offset);
    cudaFree(d_blkwidth);
    cudaFree(d_hyb_coocount);
    cudaFree(d_hyb_offset);

    cudaFree(d_Tile_csr_Val);
    cudaFree(d_Tile_csr_Col);
    cudaFree(d_Tile_csr_Ptr);
    cudaFree(d_Tile_ell_Val);
    cudaFree(d_Tile_ell_colIdx);
    cudaFree(d_Tile_hyb_Val);
    cudaFree(d_Tile_hyb_ellcolIdx);
    cudaFree(d_Tile_hyb_coorowIdx);
    cudaFree(d_Tile_dns_Val);
    cudaFree(d_Tile_dnsrow_Val);
    cudaFree(d_Tile_dnsrow_idx);
    cudaFree(d_Tile_dnscol_Val);
    cudaFree(d_Tile_dnscol_idx);
    cudaFree(d_coo_new_value);
    cudaFree(d_coo_new_rowidx);
    cudaFree(d_coo_new_matrix_ptr);
    cudaFree(d_coo_new_matrix_colidx);

printf("--------------------------------!!-cuda spmv end-!!------------------------------------\n");

    Tile_destroy(matrixA_d);
    free(new_nnz_count_1);
    free(new_coo_colidx_1);
    free(new_coo_rowidx_1);
    free(new_coo_value_1);
    free(group_ptr_1);
    cudaFree(d_new_nnz_count);
    cudaFree(d_coo_new_colidx);
    cudaFree(d_coo_new_value);
    cudaFree(d_new_num);
    cudaFree(d_new_coo_value);
    cudaFree(d_new_coo_rowidx);


    return 0;
}
