#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>


int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int  key_input,
                                        const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

void Dot_Product_Avx2_dLBL(int len,
                        const int *indx,
                        const double *Val,
                        const double *X,
                        double *res) 
{

    const int *colIndPtr = indx;
    const double *matValPtr = (double *) Val;
    const double *x = (double *) X;

    int j;
    double result = 0.0;

    __m256d vec_y;
    vec_y = _mm256_setzero_pd();
    int nnzThisLine = len;
    int k_iter = nnzThisLine / 4;
    int k_rem = nnzThisLine % 4;

    //Loop in multiples of 4 non-zeroes
    for (j = 0; j < k_iter; j++) {
        vec_y = _mm256_fmadd_pd(
                *((__m256d_u *) (matValPtr)),
                _mm256_set_pd(x[*(colIndPtr + 3)],
                              x[*(colIndPtr + 2)],
                              x[*(colIndPtr + 1)],
                              x[*(colIndPtr)]),
                vec_y);

        matValPtr += 4;
        colIndPtr += 4;
    }

    // Horizontal addition
    if (k_iter) {
        // sum[0] += sum[1] ; sum[2] += sum[3]
        vec_y = _mm256_hadd_pd(vec_y, vec_y);
        // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
        __m128d sum_lo = _mm256_castpd256_pd128(vec_y);
        // Extract 128 bits to obtain sum[2] and sum[3]
        __m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);
        // Add remaining two sums
        __m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
        // Store result
        result = sse_sum[0];
    }

    //Remainder loop for nnzThisLine%4
    for (j = 0; j < k_rem; j++)
	{
        result += *matValPtr++ * x[*colIndPtr++];
    }
    *(double *) res = result;
}

//int main(int argc, char ** argv)
int spmvLBL(int m,int n,int nnzR,int* RowPtr,int* ColIdx,double*Val,char* filename,double* GFlops_LBL,double* Time_LBL,double* time_pre,double* LBL_error)
{
	//char *filename = argv[1];
	//printf ("filename = %s\n", filename);

	//read matrix
	//int m, n, nnzR, isSymmetric;

	//mmio_info(&m, &n, &nnzR, &isSymmetric, filename);
	//int *RowPtr = (int *)malloc((m+1) * sizeof(int));
	//int *ColIdx = (int *)malloc(nnzR * sizeof(int));
	//double *Val    = (double *)malloc(nnzR * sizeof(double));
	//mmio_data(RowPtr, ColIdx, Val, filename);
	for (int i = 0; i < nnzR; i++)
	    Val[i] = 1;

	//create X, Y,Y_golden
	double *X = (double *)malloc(sizeof(double) * (n+1));
	double *Y = (double *)malloc(sizeof(double) * (m+1));
	double *Y_golden = (double *)malloc(sizeof(double) * (m+1));

	memset (X, 0, sizeof(double) * (n+1));
	memset (Y, 0, sizeof(double) * (m+1));
	memset (Y_golden, 0, sizeof(double) * (m+1));
	
	for (int i = 0; i < n; i++)
		X[i] = 1;

	for (int i = 0; i < m; i++)
	    for(int j = RowPtr[i]; j < RowPtr[i+1]; j++)
		 Y_golden[i] += Val[j] * X[ColIdx[j]];

	//int nthreads = atoi(argv[2]);
  	//omp_set_num_threads(nthreads);
	int nthreads = omp_get_max_threads();

	//int iter = atoi(argv[3]);
  	//printf("#iter is %i \n", iter);
	int iter = 500;

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// find balanced points
	int *csrSplitter = (int *)malloc((nthreads+1) * sizeof(int));
	//int *csrSplitter_normal = (int *)malloc((nthreads+1) * sizeof(int));
	int stridennz = ceil((double)nnzR/(double)nthreads);

	//#pragma omp parallel for
	for (int tid = 0; tid <= nthreads; tid++)
	{
		// compute partition boundaries by partition of size stride
		int boundary = tid * stridennz;
		// clamp partition boundaries to [0, nnzR]
		boundary = boundary > nnzR ? nnzR : boundary;
		// binary search
		csrSplitter[tid] = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;
		
	}
	csrSplitter[0] = 0;

	//#pragma omp parallel for
    for (int tid = 1; tid <= nthreads; tid++) 
	{
        // compute partition boundaries by partition of size stride
        int boundary = tid * stridennz;
        // clamp partition boundaries to [0, nnzR]
        boundary = boundary > nnzR ? nnzR : boundary;
        // binary search
        int spl = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;
        if(spl==csrSplitter[tid-1])
		{
            spl = m>(spl+1)? (spl+1):m;
            csrSplitter[tid] = spl;
        }
		else
		{
            csrSplitter[tid] = spl;
        }
    }
	gettimeofday(&t2, NULL);
	double time_balanced_pre = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	//printf("time_balanced_pre = %f\n", time_balanced_pre);
	time_pre[1] = time_balanced_pre;
	/*
//-----------------------------------parallel_omp_balanced-------------------------------------
	gettimeofday(&t1, NULL);
	int currentiter = 0;
	for (currentiter = 0; currentiter < iter; currentiter++)
	{
		#pragma omp parallel for
		for (int tid = 0; tid < nthreads; tid++)
		{
			for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
			{
				double sum = 0;
				for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++) 
				{
					sum += Val[j] * X[ColIdx[j]];
				}
				Y[u] = sum;
			}
		}
	}
	gettimeofday(&t2, NULL);
	double time_balanced = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / iter;
	double GFlops_balanced = 2 * nnzR / time_balanced / pow(10,6);
	int errorcount_balanced = 0;
	for (int i = 0; i < m; i++)
		if (Y[i] != Y_golden[i])
			errorcount_balanced++;

	//printf("time_balanced = %f\n", time_balanced);
	//printf("errorcount_balanced = %i\n", errorcount_balanced);
	//printf("GFlops_balanced = %f\n", GFlops_balanced);
	GFlops_LBL[0] = GFlops_balanced;
	Time_LBL[0] = time_balanced;
	LBL_error[0] = errorcount_balanced;
//------------------------------------------------------------------------
*/
//------------------------------------parallel_omp_balanced_avx2------------------------------------
	int currentiter = 0;
	gettimeofday(&t1, NULL);
	for (currentiter = 0; currentiter < iter; currentiter++)
	{
		#pragma omp parallel for
		for (int tid = 0; tid < nthreads; tid++)
		{
			for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
				{
					Dot_Product_Avx2_dLBL(RowPtr[u + 1] - RowPtr[u],
                                   ColIdx + RowPtr[u],
                                   Val,
                                   X,
                                   Y + u);
				}
		}
	}
	gettimeofday(&t2, NULL);
	double time_balanced_avx = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / iter;
	double GFlops_balanced_avx = 2 * nnzR / time_balanced_avx / pow(10,6);
	int errorcount_balanced_avx = 0;
	for (int i = 0; i < m; i++)
		if (Y[i] != Y_golden[i])
			errorcount_balanced_avx++;

	//printf("time_balanced_avx = %f\n", time_balanced_avx);
	//printf("errorcount_balanced_avx = %i\n", errorcount_balanced_avx);
	//printf("GFlops_balanced_avx = %f\n", GFlops_balanced_avx);
	GFlops_LBL[1] = GFlops_balanced_avx;
	Time_LBL[1] = time_balanced_avx;
	LBL_error[1] = errorcount_balanced_avx;

//------------------------------------------------------------------------
	return 0;
}

