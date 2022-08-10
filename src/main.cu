#include "common.h"
#include "mmio_highlevel.h"
#include "utils.h"
#include "csr2tile.h"
#include "external/CSR5_cuda/anonymouslib_cuda.h"
#include "tilespmv_cpu.h"
#include "tilespmv_cuda.h"

# define INDEX_DATA_TYPE unsigned char


#define DEBUG_FORMATCOST 0


int main(int argc, char ** argv)
{

	if (argc < 2)
    {
        printf("Run the code by './test matrix.mtx'.\n");
        return 0;
    }
	
    printf("--------------------------------!!!!!!!!------------------------------------\n");

 	struct timeval t1, t2;
	int rowA;
	int colA;
	MAT_PTR_TYPE nnzA;
	int isSymmetricA;
    MAT_VAL_TYPE *csrValA;
    int *csrColIdxA;
    MAT_PTR_TYPE *csrRowPtrA;
	
    int device_id = 0;
    // "Usage: ``./spmv -d 0 mtx A.mtx'' for Ax=y on device 0"
    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);


	char  *filename;
    filename = argv[3];
    printf("MAT: -------------- %s --------------\n", filename);

    // load mtx A data to the csr format
    gettimeofday(&t1, NULL);
    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtrA, &csrColIdxA, &csrValA, filename);
    gettimeofday(&t2, NULL);
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("  input matrix A: ( %i, %i ) nnz = %i\n  loadfile time    = %4.5f sec\n", rowA, colA, nnzA, time_loadmat/1000.0);

	for (int i = 0; i < nnzA; i++)
	    csrValA[i] = i % 10;

    rowA = (rowA / BLOCK_SIZE) * BLOCK_SIZE;

    // set device
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    printf("---------------------------------------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n", device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);



    Tile_matrix *matrixA = (Tile_matrix *)malloc(sizeof (Tile_matrix));

    //format conversion

    Tile_create(matrixA, 
                rowA, colA, nnzA,
                csrRowPtrA,
                csrColIdxA,
                csrValA);

	MAT_VAL_TYPE *x = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * colA);
	for (int i = 0; i < colA; i++)
	{
		x[i] = i % 10;
	}

    // compute reference results on a cpu core

	MAT_VAL_TYPE *y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
	for (int i = 0; i < rowA; i++)
	{
		MAT_VAL_TYPE sum = 0;
		for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
		{
			sum += csrValA[j] * x[csrColIdxA[j]];
		}
		y_golden[i] = sum;
	}




    //run CPU TileSpMV
    

    struct timeval cpu_tstart,cpu_tend;
    gettimeofday(&cpu_tstart, NULL);

	MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
    memset(y, 0, sizeof(MAT_VAL_TYPE) * rowA);

    int tilenum = matrixA->tilenum;


    int * ptroffset1 = (int *)malloc(sizeof(int) * tilenum);
    int * ptroffset2 = (int *)malloc(sizeof(int) * tilenum);
    memset(ptroffset1, 0, sizeof(int) * tilenum);
    memset(ptroffset2, 0, sizeof(int) * tilenum);

    int rowblkblock = 0;

    unsigned int * blkcoostylerowidx ;
    int * blkcoostylerowidx_colstart   ;
    int * blkcoostylerowidx_colstop ;
    int *multicoo_ptr = (int *)malloc((rowA + 1) * sizeof(int));

    int *multicoo_colidx ;
    MAT_VAL_TYPE *multicoo_val ;

    tilespmv_cpu(matrixA,
                ptroffset1,
                ptroffset2,
                &rowblkblock,
                &blkcoostylerowidx,
                &blkcoostylerowidx_colstart,
                &blkcoostylerowidx_colstop,
                rowA, colA, nnzA,
                csrRowPtrA,
                csrColIdxA,
                csrValA,
                x,
                y,
                y_golden
            );


  MAT_VAL_TYPE alpha = 1.0;
  memset(y, 0, sizeof(MAT_VAL_TYPE) * rowA);


//run GPU TilespMV

    call_tilespmv_cuda( filename,
                        matrixA,
                        ptroffset1,
                        ptroffset2,
                        rowblkblock,
                        blkcoostylerowidx,
                        blkcoostylerowidx_colstart,
                        blkcoostylerowidx_colstop,
                        rowA, colA, nnzA,
                        csrRowPtrA,
                        csrColIdxA,
                        csrValA,
                        alpha,
                        x,
                        y,
                        y_golden);



    //check results

    int error_count_cuda = 0;
    for (int i = 0; i < rowA; i++)
        if (abs(y_golden[i] - y[i]) > 0.01 * abs(y[i]))
        {
            error_count_cuda++;
            // cout<<"y_golden = "<<y_golden[i]<<" , "<<"y = "<<y[i]<<endl;
        }

    if (error_count_cuda == 0)
        cout << "Check... PASS!" << endl;
    else
        cout << "Check... NO PASS! error_count_cuda = "<< error_count_cuda<< endl;


    free(matrixA);
    free(csrValA);
    free(csrColIdxA);
    free(csrRowPtrA);

}
