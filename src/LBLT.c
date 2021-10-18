#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>

int binary_search_right_boundary_kernel_LBLT(const int *row_pointer,
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

void Dot_Product_Avx2_dLBLT(int len,
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
int spmvLBLT(int m,int n,int nnzR,int* RowPtr,int* ColIdx,double*Val,double* GFlops_LBLT,double* Time_LBLT,double* time_pre,double* LBLT_error)
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
	//printf("The order of the rating matrix R is %i by %i, #nonzeros = %i\n",m, n, nnzR);

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
  	//printf("#threads is %i \n", nthreads);
	int nthreads = omp_get_max_threads();
	//printf("omp_num_thread_LBL = %i\n", nthreads);

	//int iter = atoi(argv[3]);
  	//printf("#iter is %i \n", iter);
	int iter = 500;

	struct timeval t1, t2, t3;
	gettimeofday(&t1, NULL);

    int *csrSplitter_yid = (int *)malloc((nthreads+1) * sizeof(int));
	int stridennz = ceil((double)nnzR/(double)nthreads);

	//#pragma omp parallel for
	for (int tid = 0; tid <= nthreads; tid++)
	{
		// compute partition boundaries by partition of size stride
		int boundary_yid = tid * stridennz;
		// clamp partition boundaries to [0, nnzR]
		boundary_yid = boundary_yid > nnzR ? nnzR : boundary_yid;
		// binary search
		csrSplitter_yid[tid] = binary_search_right_boundary_kernel_LBLT(RowPtr, boundary_yid, m + 1) - 1;
		//printf("csrSplitter_yid[%d] is %d\n", tid, csrSplitter_yid[tid]);	
		
	}
	gettimeofday(&t2, NULL);
    int *Apinter = (int *)malloc(nthreads * sizeof(int));
    memset(Apinter, 0, nthreads *sizeof(int) );
	//每个线程执行行数
	//#pragma omp parallel for
	for (int tid = 0; tid < nthreads; tid++)
	{
		Apinter[tid] = csrSplitter_yid[tid+1] - csrSplitter_yid[tid];
		//printf("A[%d] is %d\n", tid, Apinter[tid]);	
	}

	int *Bpinter = (int *)malloc(nthreads * sizeof(int));
    memset(Bpinter, 0, nthreads *sizeof(int) );
	//每个线程执行非零元数
	//#pragma omp parallel for
	for (int tid = 0; tid < nthreads; tid++)
	{
		int num = 0;
		for (int u = csrSplitter_yid[tid]; u < csrSplitter_yid[tid+1]; u++)
		{
			num += RowPtr[ u + 1 ] - RowPtr[u];
		}
		Bpinter[tid] = num;
		//printf("B [%d]is %d\n",tid, Bpinter[tid]);
	}
	
	int *Yid = (int *)malloc(sizeof(int) * nthreads);
	memset (Yid, 0, sizeof(int) * nthreads);
	//每个线程
	int flag = -2;
	//#pragma omp parallel for
	for (int tid = 0; tid < nthreads; tid++)
	{
		//printf("tid = %i, csrSplitter: %i -> %i\n", tid, csrSplitter_yid[tid], csrSplitter_yid[tid+1]);
		if (csrSplitter_yid[tid + 1] - csrSplitter_yid[tid] == 0 && tid != 0)
		{
			Yid[tid] = csrSplitter_yid[tid];
			flag = 1;
		}
		else if (flag == 1)
		{
			Yid[tid] = csrSplitter_yid[tid];
			flag = -2;
		}
		else
		{
			Yid[tid] = -1;
		}
	}

    //行平均用在多行上
	//int sto = nthreads > nnzR ? nthreads : nnzR;
	int *Start1 = (int *)malloc(sizeof(int) * nthreads);
	memset (Start1, 0, sizeof(int) * nthreads);
	int *End1 = (int *)malloc(sizeof(int) * nthreads);
	memset (End1, 0, sizeof(int) * nthreads);
	int *label = (int *)malloc(sizeof(int) * nthreads);
	memset (label, 0, sizeof(int) * nthreads);

	int start1, search1 = 0;
	//#pragma omp parallel for
	for (int tid = 0;tid < nthreads;tid++)
	{
		if (Apinter[tid] == 0)
		{
			if(search1 == 0)
			{
				start1 = tid;
				search1 = 1; 
			}
		}
		if(search1 == 1 && Apinter[tid]!= 0)
		{
			int nntz = floor((double)Apinter[tid] / (double)(tid-start1+1));
			if( nntz != 0)
			{
				for(int i = start1;i <= tid;i++)
				{
					label[i] = i;
				}
			}
			else if((tid-start1+1) >= Apinter[tid] && Apinter[tid] != 0)
			{
				for(int i = start1;i <= tid;i++)
				{
					label[i] = i;
				}
			}
			int mntz = Apinter[tid] - (nntz * (tid-start1));
			//start and end
			int n = start1;
			Start1[n] = csrSplitter_yid[tid];
			End1[n] = Start1[n] + nntz;
			//printf("start1a[%d] = %d, end1a[%d] = %d\n",n,Start1[n],n, End1[n]);
			for (int p = start1 + 1; p <= tid ; p++)
			{
				if(p == tid)
				{
					Start1[p] = End1[p - 1];
					End1[p] = Start1[p] + mntz;
				}
				else
				{
					Start1[p] = End1[p-1];
					End1[p] = Start1[p] + nntz;
				}
				//printf("start1b[%d] = %d, end1b[%d] = %d\n",n,Start1[n],n, End1[n]);
			}
			search1 = 0;
		}
	}

	//非零元平均用在行数小于线程数
	double *Ypartialsum = (double *)malloc(sizeof(double) * nthreads);
	memset (Ypartialsum, 0, sizeof(double) * nthreads);
	double *Ysum = (double *)malloc(sizeof(double) * nthreads);
	memset (Ysum, 0, sizeof(double) * nthreads);
    int *Start2 = (int *)malloc(sizeof(int) * nthreads);
	memset (Start2, 0, sizeof(int) * nthreads);
	int *End2 = (int *)malloc(sizeof(int) * nthreads);
	memset (End2, 0, sizeof(int) * nthreads);
	int start2, search2 = 0;
	//#pragma omp parallel for
	for (int tid = 0;tid < nthreads;tid++)
	{
		if (Bpinter[tid] == 0)
		{
			if(search2 == 0)
			{
				start2 = tid;
				search2 = 1;
			}
		}
		if(search2 == 1 && Bpinter[tid]!= 0)
		{
			int nntz2 = floor((double)Bpinter[tid] / (double)(tid-start2+1));
			int mntz2 = Bpinter[tid] - (nntz2 * (tid-start2));
			//start and end
			int n = start2;
			for (int i = start2; i >= 0; i--)
			{
				Start2[n] += Bpinter[i];
				End2[n] = Start2[n] + nntz2;
			}
			for (n = start2 + 1; n < tid ; n++)
			{
				Start2[n] = End2[n-1];
				End2[n] = Start2[n] + nntz2;
			}
			if (n == tid)
			{
				Start2[n] = End2[n - 1];
				End2[n] = Start2[n] + mntz2;
			}
			search2 = 0;
	    }
	}
	gettimeofday(&t3, NULL);
	double time_LBL_pre = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	double time_LBLT_pre = ((t3.tv_sec - t1.tv_sec) * 1000.0 + (t3.tv_usec - t1.tv_usec) / 1000.0);

	for(int tid = 0; tid < nthreads; tid++)
	{
		if(Yid[tid] != -1)
		{
			time_pre[2] = time_LBLT_pre;

		}
		else
		{
			time_pre[2] = time_LBL_pre;
		}
	}

//-----------------------------------parallel_omp_balanced_Yid-------------------------------------
	int currentiter = 0;
	gettimeofday(&t1, NULL);
	for (currentiter = 0; currentiter < iter; currentiter++)
	{
		#pragma omp parallel for
		for (int tid = 0; tid < nthreads; tid++)
		    Y[Yid[tid]] = 0;
		#pragma omp parallel for
		for (int tid = 0; tid < nthreads; tid++)
		{
			if (Yid[tid] == -1)
			{
				for (int u = csrSplitter_yid[tid]; u < csrSplitter_yid[tid+1]; u++)
				{
					double sum = 0;
					for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++) 
					{
						sum += Val[j] * X[ColIdx[j]];
					}
					Y[u] = sum;
				}
			}
			if (label[tid] != 0)
			{
				for (int u = Start1[tid]; u < End1[tid]; u++)
				{
					double sum = 0;
					for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++) 
					{
						sum += Val[j] * X[ColIdx[j]];
					}
					Y[u] = sum;
				}
			}
			if (Yid[tid] != -1 && label[tid] == 0)
			{
				Ysum[tid] = 0;
				Ypartialsum[tid] = 0;
				for (int j = Start2[tid]; j < End2[tid]; j++)
				{
					Ypartialsum[tid] += Val[j] * X[ColIdx[j]];
				}
				Ysum[tid] += Ypartialsum[tid];
                Y[Yid[tid]] += Ysum[tid];
			}
		}
	}
	gettimeofday(&t2, NULL);
	double time_balanced2 = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / iter;
	double GFlops_balanced2 = 2 * nnzR / time_balanced2 / pow(10,6);
	int errorcount_balanced2 = 0;
	for (int i = 0; i < m; i++)
		if (Y[i] != Y_golden[i])
			errorcount_balanced2++;

	//printf("time_LBLT = %f\n", time_balanced2);
	//printf("errorcount_LBLT = %i\n", errorcount_balanced2);
	//printf("GFlops_balanced2 = %f\n", GFlops_balanced2);
	GFlops_LBLT[0] = GFlops_balanced2;
	Time_LBLT[0] = time_balanced2;
	LBLT_error[0] = errorcount_balanced2;
//-----------------------------------------------------------------------

//------------------------------------parallel_omp_balanced_avx2_Yid------------------------------------
	gettimeofday(&t1, NULL);
	for (currentiter = 0; currentiter < iter; currentiter++)
	{
		#pragma omp parallel for
		for (int tid = 0; tid < nthreads; tid++)
		    Y[Yid[tid]] = 0;
		#pragma omp parallel for
		for (int tid = 0; tid < nthreads; tid++) 
		{
			if (Yid[tid] == -1) 
			{
				//printf("%d %d\n",tid,csrSplitter[tid]);
				for (int u = csrSplitter_yid[tid]; u < csrSplitter_yid[tid + 1]; u++) 
				{
					Dot_Product_Avx2_dLBLT(RowPtr[u + 1] - RowPtr[u],
									ColIdx + RowPtr[u],
									Val,
                                   	X,
                                   	Y + u);
				}
				
			} 
			else if (label[tid] != 0) 
			{
				for (int u = Start1[tid]; u < End1[tid]; u++) 
				{
					Dot_Product_Avx2_dLBLT(
							RowPtr[u + 1] - RowPtr[u],
							ColIdx + RowPtr[u],
							Val,
                            X,
                            Y + u);
				}
				
			}
			if (Yid[tid] != -1 && label[tid] == 0)
			{
				Dot_Product_Avx2_dLBLT(
						End2[tid] - Start2[tid],
						ColIdx + Start2[tid],
						Val + Start2[tid],
						X,
						Ysum + tid);

			}
			
		}
	}
	gettimeofday(&t2, NULL);
	double time_balanced2_avx = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / iter;
	double GFlops_balanced2_avx = 2 * nnzR / time_balanced2_avx / pow(10,6);
	int errorcount_balanced2_avx = 0;
	for (int i = 0; i < m; i++)
		if (Y[i] != Y_golden[i])
			errorcount_balanced2_avx++;

	//printf("time_balanced2_avx = %f\n", time_balanced2_avx);
	//printf("errorcount_balanced2_avx = %i\n", errorcount_balanced2_avx);
	//printf("GFlops_balanced2_avx = %f\n", GFlops_balanced2_avx);
	GFlops_LBLT[1] = GFlops_balanced2_avx;
	Time_LBLT[1] = time_balanced2_avx;
	LBLT_error[1] = errorcount_balanced2_avx;
//------------------------------------------------------------------------
	return 0;
}

