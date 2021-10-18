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

//int main(int argc, char ** argv)
int spmvLBLT(int new_row,int nthreads,int m,int n,int nnzR,int* coo_new_rowidx,int *coo_new_matrix_ptr,
			int *coo_new_matrix_colidx,double* coo_new_matrix_value,int* csrSplitter_yid,int* Yid,
            int* Start1,int* End1,int* label,int* Start2,int* End2)
{
	
	int stridennz = ceil((double)nnzR/(double)nthreads);

	//#pragma omp parallel for
	for (int tid = 0; tid <= nthreads; tid++)
	{
		// compute partition boundaries by partition of size stride
		int boundary_yid = tid * stridennz;
		// clamp partition boundaries to [0, nnzR]
		boundary_yid = boundary_yid > nnzR ? nnzR : boundary_yid;
		// binary search
		csrSplitter_yid[tid] = binary_search_right_boundary_kernel_LBLT(coo_new_matrix_ptr, boundary_yid, new_row + 1) - 1;
		//printf("csrSplitter_yid[%d] is %d\n", tid, csrSplitter_yid[tid]);	
		
	}
	
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
			num += coo_new_matrix_ptr[ u + 1 ] - coo_new_matrix_ptr[u];
		}
		Bpinter[tid] = num;
		//printf("B [%d]is %d\n",tid, Bpinter[tid]);
	}
	
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
		//printf("Yid[%d] = %d\n",tid,Yid[tid]);
	}

    //行平均用在多行上
	//int sto = nthreads > nnzR ? nthreads : nnzR;
	memset (Start1, 0, sizeof(int) * nthreads);
	memset (End1, 0, sizeof(int) * nthreads);
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
	memset (Start2, 0, sizeof(int) * nthreads);
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
	return 0;
}

