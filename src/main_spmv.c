#include"common.h"
#include"mmio_highlevel.h"
#include"utils.h"
#include "encode.h"


# define INDEX_DATA_TYPE unsigned char
//# define VAL_DATA_TYPE double


typedef struct 
{
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
}SMatrix;



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
	SMatrix matrixA;


	char  *filename;
    filename = argv[1];
    printf("MAT: -------------- %s --------------\n", filename);

    // load mtx A data to the csr format
    gettimeofday(&t1, NULL);
    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &matrixA.rowpointer, &matrixA.columnindex, &matrixA.value, filename);
    gettimeofday(&t2, NULL);
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", rowA, colA, nnzA, time_loadmat/1000.0);

	for (int i = 0; i < nnzA; i++)
	    matrixA.value[i] = i % 10;

    if (rowA != colA)
    {
        printf("This code only computes square matrices.\n Exit.\n");
        return 0;
    }
//	rowA=16;
//	nnzA=matrixA.rowpointer[16];
//	MAT_PTR_TYPE *cscColPtrA = (MAT_PTR_TYPE *)malloc((colA+1) * sizeof(MAT_PTR_TYPE));
 //   int *cscRowIdxA = (int *)malloc(nnzA   * sizeof(int));
 //   MAT_VAL_TYPE *cscValA    = (MAT_VAL_TYPE *)malloc(nnzA  * sizeof(MAT_VAL_TYPE));


	 // transpose A from csr to csc
 //   matrix_transposition(rowA, colA, nnzA, matrixA.rowpointer, matrixA.columnindex, matrixA.value,cscRowIdxA, cscColPtrA, cscValA);

/*	SMatrix matrixB;
	int rowB=colA;
	int colB=rowA;

	matrixB.rowpointer = cscColPtrA;
    matrixB.columnindex = cscRowIdxA;
    matrixB.value    = cscValA;
*/


    if (BLOCK_SIZE>rowA){
		printf("Error!\n");
		return 0;
	}


    int rbnum=0;
    int cbnum=0;

    rbnum = rowA%BLOCK_SIZE==0 ? rowA/BLOCK_SIZE : (rowA/BLOCK_SIZE)+1 ;

    cbnum = colA%BLOCK_SIZE==0 ? colA/BLOCK_SIZE : (colA/BLOCK_SIZE)+1 ;
    
  
    char *flag=(char *)malloc(cbnum*sizeof(char));
    int nnzbl=0;

    for (int i=0;i<rbnum;i++)
	{
        memset(flag,0,cbnum*sizeof(char));
        int start= i*BLOCK_SIZE;
        int end = i==rbnum-1 ?  rowA : (i+1)*BLOCK_SIZE ;
		for (int j=matrixA.rowpointer[start];j<matrixA.rowpointer[end];j++){
            int jc=matrixA.columnindex[j]/BLOCK_SIZE;
            if (flag[jc]==0)
            {
                flag[jc]=1;
                nnzbl++;
            }
	    } 
	}
 //   printf("nnzbl=%d\n",nnzbl);

    MAT_PTR_TYPE *rowblock_ptr;    //block rowpointer of A
	int *columnid;                // block columnindex of A
	int *nnzb_A;
	int colid=0;
	rowblock_ptr=(MAT_PTR_TYPE *)malloc((rbnum+1)*sizeof(MAT_PTR_TYPE));
    columnid=(int *)malloc(nnzbl*sizeof(int));
	
    memset(rowblock_ptr,0,(rbnum+1)*sizeof(MAT_PTR_TYPE));
    int ptrA_length=0;
    for (int i=0;i<rbnum;i++)
	{
        memset(flag,0,cbnum*sizeof(char));
        int start= i*BLOCK_SIZE;
        int end = i==rbnum-1 ?  rowA : (i+1)*BLOCK_SIZE ;
		for (int j=matrixA.rowpointer[start];j<matrixA.rowpointer[end];j++)
        {
            int jc=matrixA.columnindex[j]/BLOCK_SIZE;
            if (flag[jc]==0)
            {
                flag[jc]=1;
                rowblock_ptr[i+1]++;
                columnid[colid]=jc;
                colid++;
                ptrA_length+=(end-start);
            }
	    } 
	}
    for (int i=1;i<rbnum+1;i++)
	{
		rowblock_ptr[i]+=rowblock_ptr[i-1];
	}
/*
    for(int blki =0;blki < rbnum ;blki ++)
    {
        quick_sort_key(columnid + rowblock_ptr[blki],rowblock_ptr[blki+1] - rowblock_ptr[blki]);
    }
*/

//    exclusive_scan(nnzb_A,nnzbl+1);

/*    for (int i=0;i<rbnum+1;i++)
	{
		printf("%d    ",rowblock_ptr[i]);
	}
    printf("\n");

	for (int i=0;i<nnzbl;i++)
	{
		printf("%d    ",columnid[i]);
	}
    printf("\n");
*/

    //format
    char * Format =(char *)malloc(rowblock_ptr[rbnum] * sizeof(char));
    memset(Format,0,rowblock_ptr[rbnum] * sizeof(char));


 /*   int *csrnum = (int *)malloc(rowblock_ptr[rbnum] * sizeof(int));
    int *coonum = (int *)malloc(rowblock_ptr[rbnum] * sizeof(int));
    int *ellnum = (int *)malloc(rowblock_ptr[rbnum] * sizeof(int));
    int *hybnum = (int *)malloc(rowblock_ptr[rbnum] * sizeof(int));
    int *dnsnum = (int *)malloc(rowblock_ptr[rbnum] * sizeof(int));
    int *dnsrnum = (int *)malloc(rowblock_ptr[rbnum] * sizeof(int));
    int *dnscnum = (int *)malloc(rowblock_ptr[rbnum] * sizeof(int));

    memset(csrnum,0,rowblock_ptr[rbnum] * sizeof(int));
    memset(coonum,0,rowblock_ptr[rbnum] * sizeof(int));
    memset(ellnum,0,rowblock_ptr[rbnum] * sizeof(int));
    memset(hybnum,0,rowblock_ptr[rbnum] * sizeof(int));
    memset(dnsnum,0,rowblock_ptr[rbnum] * sizeof(int));
    memset(dnsrnum,0,rowblock_ptr[rbnum] * sizeof(int));
    memset(dnscnum,0,rowblock_ptr[rbnum] * sizeof(int));
*/

    //nnz
    int *blknnz = (int *)malloc((nnzbl+1)*sizeof(int));



    nnzb_A=(int *)malloc((nnzbl+1)*sizeof(int));
    int nnzid=0;
 
    //dense 
    int dense_size=0;

    //denserow
    int * denserowptr = (int *)malloc((rowblock_ptr[rbnum] + 1) * sizeof(int));
    memset(denserowptr,0,(rowblock_ptr[rbnum]+ 1) * sizeof(int));
    int denserow_size =0 ;
    int denrowblknum =0 ;
//    int denrowsum =0;

    //densecolumn
    int * densecolptr = (int *)malloc((rowblock_ptr[rbnum] + 1) * sizeof(int));
    memset(densecolptr,0,(rowblock_ptr[rbnum]+ 1) * sizeof(int));
    int densecol_size =0 ;
    int dencolblknum =0 ;
//    int dencolsum =0;


    //CSR
    int csrsize=0;
    int csrptrlen=0;
//    int csrblknum=0;

    //ELL
    int ellsize =0;
//    int ellblknum=0;

    //COO
    int coosize =0;
//    int cooblknum=0;

    //HYB
    int hybellsize =0;
    int hybcoosize =0;
    int hybsize =0;
    int hybblknum =0;
    char *blkwidth = (char *)malloc(rowblock_ptr[rbnum]*sizeof(char));
    memset(blkwidth,0,rowblock_ptr[rbnum]) ;

    int *hybcoo= (int *)malloc(rowblock_ptr[rbnum]*sizeof(int));
    memset(hybcoo,0,rowblock_ptr[rbnum]) ;
    int hybblk=0;
//    int *blkhybcoonum = (int *)malloc(rowblock_ptr[rbnum]*sizeof(int));
//    memset(blkhybcoonum,0,rowblock_ptr[rbnum]) ;


    int *multicoonnz = (int * )malloc((rbnum +1) *sizeof(int));
    memset(multicoonnz,0,(rbnum +1) *sizeof(int));


    
  gettimeofday(&t1, NULL);
 //#pragma omp parallel for   
 

    for (int blki=0;blki<rbnum;blki++)
    {
        int rowbnum=rowblock_ptr[blki+1]-rowblock_ptr[blki];
        int *rownnzA=(int *)malloc((rowbnum + 1) * sizeof(int));
        memset(rownnzA,0,(rowbnum + 1)*sizeof(int));
    //    SMatrix *subrowmatrixA=(SMatrix *)malloc(rowbnum*sizeof(SMatrix));
        int rowlength= blki==rbnum-1 ? rowA-(rbnum-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int start= blki*BLOCK_SIZE;
        int end = blki==rbnum-1 ?  rowA : (blki+1)*BLOCK_SIZE ;
		for (int j=matrixA.rowpointer[start];j<matrixA.rowpointer[end];j++)
        {
            int ki;
            for (int k=rowblock_ptr[blki],ki=0;k<rowblock_ptr[blki+1],ki<rowbnum;k++,ki++)
            {
                int kcstart=columnid[k]*BLOCK_SIZE;
                int kcend= columnid[k]== (cbnum-1) ?  colA: (columnid[k]+1)*BLOCK_SIZE;
                if (matrixA.columnindex[j]>=kcstart&&matrixA.columnindex[j]<kcend)
                {
                    rownnzA[ki]++;
                    break;
                }
		    }
	    }
        int nnzsum= 0;

        for(int bi=0;bi<rowbnum;bi++)
        {
            nnzb_A[rowblock_ptr[blki]+bi]=rownnzA[bi];
            nnzsum += rownnzA[bi] ;
        //    printf("the %d's nnz=%d\n",rowblock_ptr[blki]+bi,nnzb_A[nnzid]);
         //   nnzid++;
        }

        exclusive_scan(rownnzA,rowbnum+1);

 
    //    printf("blki=%d,rowbnum=%d\n",blki,rowbnum) ;
        MAT_PTR_TYPE *rowptr = (MAT_PTR_TYPE *)malloc((rowbnum * (rowlength+1)) *sizeof(MAT_PTR_TYPE));
        memset(rowptr,0,(rowbnum * (rowlength+1))*sizeof(MAT_PTR_TYPE));
    /*    for (int bi=0;bi<rowbnum;bi++) 
        {
           subrowmatrixA[bi].rowpointer=(MAT_PTR_TYPE *)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
           memset(subrowmatrixA[bi].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
        }
    */
        unsigned char *rbcol = (char *)malloc(nnzsum *sizeof(char)) ;
        memset(rbcol , 0, nnzsum *sizeof(char));

        int *num=(int*)malloc((rowbnum)*sizeof(int));
	
	    memset(num,0,(rowbnum)*sizeof(int));

        for (int ri=0;ri<rowlength;ri++)
        {
            for (int j=matrixA.rowpointer[start+ri];j<matrixA.rowpointer[start+ri+1];j++)
            {
                int ki;
                for (int k=rowblock_ptr[blki],ki=0;k<rowblock_ptr[blki+1],ki<rowbnum;k++,ki++)
                {
                    int kcstart=columnid[k]*BLOCK_SIZE;
                    int kcend= columnid[k]== (cbnum-1) ?  colA: (columnid[k]+1)*BLOCK_SIZE;
                    if (matrixA.columnindex[j]>=kcstart&&matrixA.columnindex[j]<kcend)
                    {
                        num[ki]++;
                        rbcol[rownnzA[ki]+num[ki]-1] = matrixA.columnindex[j]-columnid[k]*BLOCK_SIZE;
                        break;
                    }
                }
            }
            for (int bi=0;bi<rowbnum;bi++)
            {
                rowptr[bi * (rowlength + 1) + ri +1]= num[bi]; 
            //    subrowmatrixA[bi].rowpointer[ri+1]=num[bi];
            }   
	    }

        char * denserowflag = (char *)malloc(rowbnum * sizeof(char));
        char * densecolflag = (char *)malloc(rowbnum * sizeof(char));

        char * samecolcount = (char *)malloc (rowlength * sizeof(char));

     
        
        for (int bi=0;bi<rowbnum;bi++)
        {
        /*    if(1)
            {
                Format[rowblock_ptr[blki]+bi] =0 ;
                csrsize += nnzb_A[rowblock_ptr[blki]+bi] ;
                blknnz[rowblock_ptr[blki] + bi] = nnzb_A[rowblock_ptr[blki]+bi] ;
                csrptrlen += rowlength ;
                continue;
            }
        */
        /*    int blkcol = columnid[rowblock_ptr[blki]+bi] ;

            if (blki == blkcol)  //CSR
            {
                Format[rowblock_ptr[blki]+bi] =0 ;
                csrsize += nnzb_A[rowblock_ptr[blki]+bi] ;
                blknnz[rowblock_ptr[blki] + bi] = nnzb_A[rowblock_ptr[blki]+bi] ;
                csrptrlen += rowlength ;
            }
*/
        //    else
            {
                
          
            
          
            int collength = columnid[rowblock_ptr[blki]+bi] == cbnum-1 ? colA - (cbnum-1 ) * BLOCK_SIZE : BLOCK_SIZE ;

            int nnzthreshold = rowlength * collength * 0.5 ;
            if (nnzb_A[rowblock_ptr[blki]+bi] >= nnzthreshold)  //dense
            {
                Format[rowblock_ptr[blki]+bi] = 4 ;
                blknnz[rowblock_ptr[blki]+bi] = rowlength * collength;
            //    dense_size += rowlength * collength ;
            //    dnsnum[rowblock_ptr[blki]+bi] = rowlength * collength;
                continue;

            }
        
            else
            {
                int iocsr_size = nnzb_A[rowblock_ptr[blki]+bi] % 2 ==0 ? nnzb_A[rowblock_ptr[blki]+bi] * sizeof (MAT_VAL_TYPE) + (nnzb_A[rowblock_ptr[blki]+bi] * sizeof (char)) / 2  + (rowlength+1) * sizeof (int) :
                                                                    nnzb_A[rowblock_ptr[blki]+bi] * sizeof (MAT_VAL_TYPE) + (nnzb_A[rowblock_ptr[blki]+bi] * sizeof (char)) /2 + 1 + (rowlength+1) * sizeof (int)  ;
            //    int iodense_sise= rowlength * collength * sizeof (MAT_VAL_TYPE) ;

            //    int minsize= iocsr_size >= iodense_sise ? iodense_sise : iocsr_size ;

                int denserownum =0;
                int densecolnum =0;

                if (nnzb_A[rowblock_ptr[blki]+bi] % collength ==0)
                {
                    int dnsrowflag =0 ;
                    int dnscolflag =0;
                    for (int ri=0;ri < rowlength ;ri++)
                    {
                        if ((rowptr[bi * (rowlength + 1)+ ri +1] - rowptr[bi * (rowlength + 1)+ ri ]) % collength !=0)
                    //    if ((subrowmatrixA[bi].rowpointer[ri+1] - subrowmatrixA[bi].rowpointer[ri] ) % collength !=0 )
                        {
                            dnsrowflag =0;
                        //    denserowflag[bi]=0;
                            break;
                        }
                        else 
                        {
                            if ((rowptr[bi * (rowlength + 1)+ ri +1] - rowptr[bi * (rowlength + 1)+ ri ])  ==collength)
                        //    if (subrowmatrixA[bi].rowpointer[ri+1] - subrowmatrixA[bi].rowpointer[ri] == collength)
                            {
                                 dnsrowflag =1;
                             //   denserowflag[bi]=1;
                                denserownum ++ ;
                            }
                        }
                        
                    }
                    //dense row
                    if (dnsrowflag  == 1)
                    {                    
                        Format[rowblock_ptr[blki]+bi] = 5 ;   //Dense Row
                    //    denrowsum += denserownum ;
                        denserowptr[rowblock_ptr[blki]+bi] = denserownum ;
                    //    denserow_size += denserownum * collength ;
                    //    dnsrnum[rowblock_ptr[blki]+bi] = denserownum * collength;

                        blknnz[rowblock_ptr[blki]+bi] = denserownum * collength;
                        continue;
                     //    break;
                    //    denrowblknum ++ ;
                    }
                    //dense column
                    else
                    {
                        memset(samecolcount,0,rowlength * sizeof(char)) ;
                        for (int ci=rownnzA[bi];ci<rownnzA[bi+1];ci++)
                        {
                            for (int colid=0 ;colid <rowlength ;colid ++)
                            {
                                if (rbcol[ci]==colid)
                                {
                                    samecolcount[colid] ++;
                                }
                            }
                        }
                        for (int ci=0;ci<rowlength ;ci ++)
                        {
                            if (samecolcount[ci] % rowlength !=0)
                            {
                                densecolflag[bi] =0;
                                break ;
                            }
                            else if (samecolcount[ci] ==rowlength)
                            {
                                densecolflag[bi]=1;
                                densecolnum ++ ;
                            }
                        }
                        if (densecolflag[bi] == 1)
                        {                    
                            Format[rowblock_ptr[blki]+bi] = 6 ;   //Dense column
                        //    dencolsum += densecolnum ;
                            densecolptr[rowblock_ptr[blki]+bi] = densecolnum ;
                        //    densecol_size += densecolnum * rowlength ;
                            blknnz[rowblock_ptr[blki]+bi] = densecolnum * rowlength;
                             continue;;
                        //    denrowblknum ++ ;
                        }
                    
                    }
                    
                }
            
                if (Format[rowblock_ptr[blki]+bi] != 5 && Format[rowblock_ptr[blki]+bi] != 6)
                {

                    int bwidth=0;
                    int hybwidth=0;
                    for (int blkj=0;blkj<rowlength;blkj++)
                    {
                        if (bwidth < rowptr[bi * (rowlength+1) +blkj+1] - rowptr[bi * (rowlength+1) +blkj])
                    //    if (bwidth < subrowmatrixA[bi].rowpointer[blkj+1]-subrowmatrixA[bi].rowpointer[blkj])
                    //        bwidth = subrowmatrixA[bi].rowpointer[blkj+1]-subrowmatrixA[bi].rowpointer[blkj];
                            bwidth = rowptr[bi * (rowlength+1) +blkj+1] - rowptr[bi * (rowlength+1) +blkj] ;
                    }
                    
                    if (nnzb_A[rowblock_ptr[blki] + bi] <= 12 )
                    {
                    /*    if (bwidth <= 2)   //ELL
                        {
                            Format[rowblock_ptr[blki]+bi] = 2;
                            ellsize += bwidth * rowlength ;
                            blkwidth[rowblock_ptr[blki]+bi]=bwidth;
                            blknnz[rowblock_ptr[blki]+bi] = bwidth * rowlength ;

                        }
                        else   //COO
                    */
                        {
                            Format[rowblock_ptr[blki]+bi] = 1;
                        //    coosize += nnzb_A[rowblock_ptr[blki]+bi] ;
                            multicoonnz[blki] += nnzb_A[rowblock_ptr[blki]+bi] ;
                            blknnz[rowblock_ptr[blki] + bi] = nnzb_A[rowblock_ptr[blki]+bi] ;
                            continue;

                        }
                    }

                    else
                    {
                        double row_length_mean = ((double)nnzb_A[rowblock_ptr[blki] + bi]) / rowlength;
                        double variance             = 0.0;
                        double row_length_skewness   = 0.0;

                        for (int row = 0; row < rowlength; ++row)
                        {
                            int length              = rowptr[bi * (rowlength + 1)+row + 1] - rowptr[bi * (rowlength + 1)+row ] ;
                            double delta                = (double)(length - row_length_mean);
                            variance   += (delta * delta);
                            row_length_skewness   += (delta * delta * delta);
                        }
                        variance                    /= rowlength;
                        double row_length_std_dev    = sqrt(variance);
                        row_length_skewness   = (row_length_skewness / rowlength) / pow(row_length_std_dev, 3.0);
                        double row_length_variation  = row_length_std_dev / row_length_mean;

                    /*    if (row_length_variation >=1)
                        {
                            printf("row_length_variation = %f\n",row_length_variation);
                        }
                    */

                        double ell_csr_threshold = 0.2;
                        double csr_hyb_threshold =  1;

                     //   printf("row_length_variation = %f\n",row_length_variation);
                        if (row_length_variation <= ell_csr_threshold)  //ELL
                        {
                            Format[rowblock_ptr[blki]+bi] = 2;
                        //    ellsize += bwidth * rowlength ;
                            blkwidth[rowblock_ptr[blki]+bi]=bwidth;
                            blknnz[rowblock_ptr[blki]+bi] = bwidth * rowlength ;
                            continue;

                        }

                        else 
                        {
                            int iopriorsize=  (bwidth * rowlength %2) ==0 ? bwidth * rowlength * sizeof (MAT_VAL_TYPE) + bwidth * rowlength * sizeof (char) /2 :
                                                                    bwidth * rowlength * sizeof (MAT_VAL_TYPE) + bwidth * rowlength * sizeof (char) /2 +1 ;
                            int ionextsize;
                            int coonextnum=0;
                            int coopriornum=0;
                            for (int wi=bwidth-1;wi>=0;wi--)
                            {
                                coonextnum=0;
                                for (int blkj=0;blkj<rowlength;blkj++)
                                {
                                if (rowptr[bi * (rowlength+1) +blkj+1] - rowptr[bi * (rowlength+1) +blkj] > wi) 
                                //   if (subrowmatrixA[bi].rowpointer[blkj+1]-subrowmatrixA[bi].rowpointer[blkj]>wi)
                                    {
                                    coonextnum += rowptr[bi * (rowlength+1) +blkj+1] - rowptr[bi * (rowlength+1) +blkj] - wi ;
                                //        coonextnum+=subrowmatrixA[bi].rowpointer[blkj+1]-subrowmatrixA[bi].rowpointer[blkj]-wi;                   
                                    } 
                                }
                                ionextsize= (wi * rowlength) % 2 ==0 ?  wi * rowlength * sizeof (MAT_VAL_TYPE )+  wi * rowlength * sizeof (char) /2 + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (char)) :
                                                                        wi * rowlength * sizeof (MAT_VAL_TYPE )+  wi * rowlength * sizeof (char) /2 + 1 + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (char)) ;
                            
                                if (iopriorsize<=ionextsize)
                                {
                                    hybwidth=wi+1;
                                    break;
                                }
                                else
                                {
                                    iopriorsize=ionextsize;
                                    coopriornum=coonextnum;
                                }

                            }

                            if (row_length_variation >= csr_hyb_threshold && coopriornum <= 12)  //HYB
                            {
                                Format[rowblock_ptr[blki]+bi] = 3;
                             //   hybcoosize+=coopriornum;
                             //   hybellsize += hybwidth * rowlength ;
                                hybcoo[rowblock_ptr[blki]+bi] = coopriornum;
                                blkwidth[rowblock_ptr[blki]+bi]=hybwidth;
                                blknnz[rowblock_ptr[blki]+bi] = coopriornum + hybwidth * rowlength ;
                                continue;

                            }
                            else   //CSR
                            {
                                Format[rowblock_ptr[blki]+bi] =0 ;
                             //   csrsize += nnzb_A[rowblock_ptr[blki]+bi] ;
                                blknnz[rowblock_ptr[blki] + bi] = nnzb_A[rowblock_ptr[blki]+bi] ;
                                continue;
                              //  csrptrlen += rowlength ;
                            }
                        }

                       
                    }
                    

                
                /*    else 
                    {
                        int iopriorsize=  (bwidth * rowlength %2) ==0 ? bwidth * rowlength * sizeof (MAT_VAL_TYPE) + bwidth * rowlength * sizeof (char) /2 :
                                                                    bwidth * rowlength * sizeof (MAT_VAL_TYPE) + bwidth * rowlength * sizeof (char) /2 +1 ;
                        int ionextsize;
                        int coonextnum=0;
                        int coopriornum=0;
                        for (int wi=bwidth-1;wi>=0;wi--)
                        {
                            coonextnum=0;
                            for (int blkj=0;blkj<rowlength;blkj++)
                            {
                            if (rowptr[bi * (rowlength+1) +blkj+1] - rowptr[bi * (rowlength+1) +blkj] > wi) 
                            //   if (subrowmatrixA[bi].rowpointer[blkj+1]-subrowmatrixA[bi].rowpointer[blkj]>wi)
                                {
                                coonextnum += rowptr[bi * (rowlength+1) +blkj+1] - rowptr[bi * (rowlength+1) +blkj] - wi ;
                            //        coonextnum+=subrowmatrixA[bi].rowpointer[blkj+1]-subrowmatrixA[bi].rowpointer[blkj]-wi;                   
                                } 
                            }
                            ionextsize= (wi * rowlength) % 2 ==0 ?  wi * rowlength * sizeof (MAT_VAL_TYPE )+  wi * rowlength * sizeof (char) /2 + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (char)) :
                                                                    wi * rowlength * sizeof (MAT_VAL_TYPE )+  wi * rowlength * sizeof (char) /2 + 1 + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (char)) ;
                        
                            if (iopriorsize<=ionextsize)
                            {
                                hybwidth=wi+1;
                                break;
                            }
                            else
                            {
                                iopriorsize=ionextsize;
                                coopriornum=coonextnum;
                            }

                        }

                    /*    if (hybwidth = 0 || iocsr_size <= iopriorsize)  //CSR
                        {
                            Format[rowblock_ptr[blki]+bi] =0 ;
                            csrsize += nnzb_A[rowblock_ptr[blki]+bi] ;
                            blknnz[rowblock_ptr[blki] + bi] = nnzb_A[rowblock_ptr[blki]+bi] ;
                            csrptrlen += rowlength ;
                        }
                       
                        if (iocsr_size <= iopriorsize)   //CSR 
                        {
                
                            Format[rowblock_ptr[blki]+bi] =0 ;
                            csrsize += nnzb_A[rowblock_ptr[blki]+bi] ;
                            blknnz[rowblock_ptr[blki] + bi] = nnzb_A[rowblock_ptr[blki]+bi] ;
                            csrptrlen += rowlength ;
                        //     break;
                
                        }
                    
                        else
                        {
                        
                        
                        /*    if (hybwidth == bwidth)  //ELL
                            {
                                Format[rowblock_ptr[blki]+bi] = 2;
                                ellsize += bwidth * rowlength ;
                                blkwidth[rowblock_ptr[blki]+bi]=bwidth;
                                blknnz[rowblock_ptr[blki]+bi] = bwidth * rowlength ;
                            //     break;
                            //    ellblknum ++ ;

                            }
                        
                            else if (coopriornum <= 10)  //HYB
                            {
                                Format[rowblock_ptr[blki]+bi] = 3;
                                hybcoosize+=coopriornum;
                                hybellsize += hybwidth * rowlength ;
                                blkwidth[rowblock_ptr[blki]+bi]=hybwidth;
                                blknnz[rowblock_ptr[blki]+bi] = coopriornum + hybwidth * rowlength ;
                            //    break;
                            //    hybblknum ++ ;
                            }
                            else
                        
                            //    if (hybwidth == 0)  //CSR
                            {
                                Format[rowblock_ptr[blki]+bi] =0 ;
                                csrsize += nnzb_A[rowblock_ptr[blki]+bi] ;
                                blknnz[rowblock_ptr[blki] + bi] = nnzb_A[rowblock_ptr[blki]+bi] ;
                                csrptrlen += rowlength ;


                            //    printf("format =1\n");
                            //    Format[rowblock_ptr[blki]+bi] = 1;
                            //    coosize += nnzb_A[rowblock_ptr[blki]+bi] ;
                            //    blknnz[rowblock_ptr[blki] + bi] = nnzb_A[rowblock_ptr[blki]+bi] ;
                            //     break;

                            }
                            
                          
                        
                        }


                    }
            */

                
            }

        }
        }
        }
        free(samecolcount);

        free(rownnzA);
    /*    for (int bi=0;bi<rowbnum;bi++)
        {
            free(subrowmatrixA[bi].rowpointer);
        }
    */
        free(rowptr);
    //    free(subrowmatrixA);
        free(num);
        free(rbcol) ;
    }




    gettimeofday(&t2, NULL);
   // printf("t1=%f,t2=%f\n",t1,t2);
    double time_transstep1  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("transform_step1 runtime    = %4.5f sec\n", time_transstep1/1000.0);

for (int blki=0;blki<rbnum;blki++)
{
    int rowlength= blki==rbnum-1 ? rowA-(rbnum-1)*BLOCK_SIZE : BLOCK_SIZE ;
    int rowbnum=rowblock_ptr[blki+1]-rowblock_ptr[blki];
    for (int bi=0;bi<rowbnum;bi++)
    {
        char format= Format[rowblock_ptr[blki]+bi];
        switch (format)
        {
            case 0:    //csr
                csrsize +=  blknnz[rowblock_ptr[blki]+bi];
                csrptrlen += rowlength ;
                break;
            
            case 1:  //coo
                coosize += blknnz[rowblock_ptr[blki]+bi];
                break;
            case 2:  //ell
                ellsize += blknnz[rowblock_ptr[blki]+bi] ;
                break;
            case 3: //hyb
                hybsize += blknnz[rowblock_ptr[blki]+bi];
                hybellsize += blkwidth[rowblock_ptr[blki]+bi] * rowlength;
                break;
            case 4:
                dense_size += blknnz[rowblock_ptr[blki]+bi];
                break;
            case 5:
                denserow_size += blknnz[rowblock_ptr[blki]+bi];
                break;
            case 6:
                densecol_size += blknnz[rowblock_ptr[blki]+bi];
                break;
        
            default:
                break;
        }

    }

}   
   
    
    for(int i=0;i<rowblock_ptr[rbnum];i++)
    {
        hybcoosize += hybcoo[i];
    }
    

    exclusive_scan(denserowptr,rowblock_ptr[rbnum]+1);
    exclusive_scan(densecolptr,rowblock_ptr[rbnum]+1);

    exclusive_scan(multicoonnz,rbnum +1);

 /*   for (int i=0;i<rowblock_ptr[rbnum]+1 ;i++)
    {
        printf("%i  ,   ",denserowptr[i]);
    }
    printf("\n") ;
*/
    exclusive_scan(blknnz,(nnzbl+1));

    int *formatnum = (int *)malloc(7 * sizeof(int));
    memset(formatnum,0,7 * sizeof(int));

    for (int j=0;j<7;j++)
    {
        for (int i=0;i<rowblock_ptr[rbnum];i++)
        {
            if (Format[i]==j)
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



//for (int i = 0; i < rowblock_ptr[rbnum]; i++)
//{
//    printf ("nnz= %i\n",nnzb_A[i]);
//}

    
   
/*    for (int i=0;i<nnzbl+1;i++)
    {
        printf("%d   ",blknnz[i]);
    }
    printf("\n");
*/
 /*for (int i=0;i<rowblock_ptr[rbnum] + 1;i++)
    {
     //   if (Format[i]==0)
     //   {
            printf("%d   ",densecolptr[i]);
     //   }
    }
     printf("\n");
*/
    
//    printf("sum=%d\n",csrsize + coosize + ellsize + hybellsize + hybcoosize + denserow_size);
//    MAT_VAL_TYPE *Block_Val=(MAT_VAL_TYPE*)malloc((csrsize + coosize + ellsize + hybellsize + hybcoosize + + dense_size + denserow_size)*sizeof(MAT_VAL_TYPE));
//    memset(Block_Val,0,(csrsize + coosize + ellsize + hybellsize + hybcoosize + denserow_size)*sizeof(MAT_VAL_TYPE));


    //CSR
    MAT_VAL_TYPE *Blockcsr_Val=(MAT_VAL_TYPE*)malloc((csrsize)*sizeof(MAT_VAL_TYPE));
	unsigned char  *Blockcsr_Col=(char*)malloc((csrsize)*sizeof(char));
	unsigned char *Blockcsr_Ptr=(char*)malloc((csrptrlen)*sizeof(char));

    int csrvid=0;
    int csrpid=0;

    //COO
    MAT_VAL_TYPE *Blockcoo_Val=(MAT_VAL_TYPE*)malloc((coosize)*sizeof(MAT_VAL_TYPE));
    unsigned char *coo_colIdx=(char*)malloc((coosize)*sizeof(char));
    unsigned char *coo_rowIdx=(char*)malloc((coosize)*sizeof(char));

    int coovid=0;
    int cooridxid=0 ;

    //ELL
    MAT_VAL_TYPE *Blockell_Val=(MAT_VAL_TYPE*)malloc((ellsize)*sizeof(MAT_VAL_TYPE));
    memset(Blockell_Val,0,(ellsize)*sizeof(MAT_VAL_TYPE));
    unsigned char *ell_colIdx=(char*)malloc((ellsize)*sizeof(char));
    memset(ell_colIdx, 0, sizeof(INDEX_DATA_TYPE) * ellsize);

//    unsigned char * ellblkwidth = (char *)malloc (ellblknum * sizeof(char));
//    int ellblk=0;
    int elloffset =0;
  //  memset(Blockell_Val, 0, sizeof(MAT_VAL_TYPE) * ellsize);

    //HYB
    MAT_VAL_TYPE *Blockhyb_Val=(MAT_VAL_TYPE*)malloc((hybellsize+hybcoosize)*sizeof(MAT_VAL_TYPE));
    memset(Blockhyb_Val,0,(hybellsize+hybcoosize)*sizeof(MAT_VAL_TYPE));
    unsigned char *hyb_ellcolIdx=(char*)malloc((hybellsize+hybcoosize)*sizeof(char));
    unsigned char * hyb_coorowIdx=(char*)malloc((hybcoosize)*sizeof(char)) ;
    memset(hyb_ellcolIdx, 0, sizeof(INDEX_DATA_TYPE) * (hybellsize+hybcoosize));
    int hyboffset =0;
    int hybcoonnzsum =0;

    //dense
    MAT_VAL_TYPE *Blockdense_Val=(MAT_VAL_TYPE*)malloc((dense_size)*sizeof(MAT_VAL_TYPE));
    memset(Blockdense_Val,0,dense_size * sizeof(MAT_VAL_TYPE));
    int denseoffset =0;
 
    //denserow
    //    printf("denserow_size=%d\n",denserow_size);
    MAT_VAL_TYPE *Blockdenserow_Val=(MAT_VAL_TYPE*)malloc((denserow_size) * sizeof(MAT_VAL_TYPE));
//    char *denserowid = (char *)malloc(denrowsum * sizeof(char));
    char *denserowid = (char *)malloc(denserowptr[rowblock_ptr[rbnum]] * sizeof(char));
    int densrow_rid= 0;
    int denserow_vid=0;

//    int * denserowptr = (int *)malloc((denrowblknum + 1) * sizeof(int));
//    memset(denserowptr,0,(denrowblknum + 1) * sizeof(int));
    int denrowcount =0;


    //dense column
    MAT_VAL_TYPE *Blockdensecol_Val=(MAT_VAL_TYPE*)malloc((densecol_size) * sizeof(MAT_VAL_TYPE));
 //   char *densecolid = (char *)malloc(dencolsum * sizeof(char));
    char *densecolid = (char *)malloc(densecolptr[rowblock_ptr[rbnum]] * sizeof(char));
    int densecol_cid= 0;
 //   int densecol_vid=0;
    int dencoloffset =0;

//    int * densecolptr = (int *)malloc((dencolblknum + 1) * sizeof(int));
//    memset(densecolptr,0,(dencolblknum + 1) * sizeof(int));
    int dencolcount =0;



     gettimeofday(&t1, NULL);
    //for each row block
    for (int blki=0;blki<rbnum;blki++)
	{
        int rowbnum=rowblock_ptr[blki+1]-rowblock_ptr[blki];
        SMatrix *subrowmatrixA=(SMatrix *)malloc(rowbnum*sizeof(SMatrix));
        int rowlength= blki==rbnum-1 ? rowA-(rbnum-1)*BLOCK_SIZE : BLOCK_SIZE ;
     //   printf("rowlength=%d\n",rowlength);
        int start= blki*BLOCK_SIZE;
        int end = blki==rbnum-1 ?  rowA : (blki+1)*BLOCK_SIZE ;
    
        for (int bi=0;bi<rowbnum;bi++) 
        {
           subrowmatrixA[bi].value=(MAT_VAL_TYPE*)malloc((nnzb_A[rowblock_ptr[blki]+bi])*sizeof(MAT_VAL_TYPE));
           subrowmatrixA[bi].columnindex=(int *)malloc((nnzb_A[rowblock_ptr[blki]+bi])*sizeof(int));
          
           subrowmatrixA[bi].rowpointer=(MAT_PTR_TYPE *)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
           memset(subrowmatrixA[bi].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
        }
        int *num=(int*)malloc((rowbnum)*sizeof(int));
	    memset(num,0,(rowbnum)*sizeof(int));

        for (int ri=0;ri<rowlength;ri++)
        {
            for (int j=matrixA.rowpointer[start+ri];j<matrixA.rowpointer[start+ri+1];j++)
            {
                int ki;
                for (int k=rowblock_ptr[blki],ki=0;k<rowblock_ptr[blki+1],ki<rowbnum;k++,ki++)
                {
                    int kcstart=columnid[k]*BLOCK_SIZE;
                    int kcend= columnid[k]== (cbnum-1) ?  colA: (columnid[k]+1)*BLOCK_SIZE;
                    if (matrixA.columnindex[j]>=kcstart&&matrixA.columnindex[j]<kcend)
                    {
                        num[ki]++;
                        subrowmatrixA[ki].value[num[ki]-1]=matrixA.value[j];
                        subrowmatrixA[ki].columnindex[num[ki]-1]=matrixA.columnindex[j]-columnid[k]*BLOCK_SIZE;
                        break; 
                    }
                }
            }
            for (int bi=0;bi<rowbnum;bi++){
                    subrowmatrixA[bi].rowpointer[ri+1]=num[bi];
            }   
	    }
    
    
        for(int bi=0;bi<rowbnum;bi++)
        {
         /*   for (int kk=0;kk<blknnz[rowblock_ptr[blki]+bi + 1] - blknnz[rowblock_ptr[blki]+bi ]; kk++ )
            {
                printf("%d    ",(int)subrowmatrixA[bi].value[kk]);
            }
            printf("\n") ;
             printf("\n") ;
        */
            int collength = columnid[rowblock_ptr[blki]+bi] == cbnum-1 ? colA - (cbnum-1 ) * BLOCK_SIZE : BLOCK_SIZE ;

        //CSR
            if (Format[rowblock_ptr[blki]+bi] == 0)
            {
                //CSR val&col
                for (int k=0;k<blknnz[rowblock_ptr[blki] + bi + 1]-blknnz[rowblock_ptr[blki] + bi];k++)
                {
                    Blockcsr_Val[csrvid]=subrowmatrixA[bi].value[k] ;
                //    Block_Val [blknnz[rowblock_ptr[blki] + bi] + k] = subrowmatrixA[bi].value[k] ;
                    Blockcsr_Col[csrvid]=subrowmatrixA[bi].columnindex[k];
                    csrvid++;
                }
                //CSR ptr
                for (int jid=0;jid<rowlength;jid++)
                {
                    Blockcsr_Ptr[csrpid]=subrowmatrixA[bi].rowpointer[jid];
                    csrpid++;
                }
            }
            //COO
            else if (Format[rowblock_ptr[blki]+bi] == 1)
            {
             /*   for (int ri = 0; ri < rowlength; ri++)
                {
                    for (int j = subrowmatrixA[bi].rowpointer[ri]; j < subrowmatrixA[bi].rowpointer[ri+1]; j++)
                    {
                        coo_rowIdx[cooridxid] = ri;
                        cooridxid++;
                        Blockcoo_Val[coovid] = subrowmatrixA[bi].value[j] ;
                    //  Block_Val [blknnz[rowblock_ptr[blki] + bi] + k] = subrowmatrixA[bi].value[k] ;
                        coo_colIdx[coovid]=subrowmatrixA[bi].columnindex[j];
                        coovid++;

                    }
                }
            */

                for (int k=0;k<blknnz[rowblock_ptr[blki] + bi + 1]-blknnz[rowblock_ptr[blki] + bi];k++)
                {  
                    Blockcoo_Val[coovid] = subrowmatrixA[bi].value[k] ;
                //    Block_Val [blknnz[rowblock_ptr[blki] + bi] + k] = subrowmatrixA[bi].value[k] ;
                    coo_colIdx[coovid]=subrowmatrixA[bi].columnindex[k];
                    coovid++;
                }
                //COO rowidx
                for (int ri = 0; ri < rowlength; ri++)
                {
                    for (int j = subrowmatrixA[bi].rowpointer[ri]; j < subrowmatrixA[bi].rowpointer[ri+1]; j++)
                    {
                        coo_rowIdx[cooridxid] = ri;
                        cooridxid++;
                    }
                }
            
            }
            //ELL col first
            else if (Format[rowblock_ptr[blki]+bi] == 2)
            {
                int bwidth=0;
                for (int bj=0;bj<rowlength;bj++)
                {
                    if (bwidth<subrowmatrixA[bi].rowpointer[bj+1]-subrowmatrixA[bi].rowpointer[bj])
                    {
                        bwidth=subrowmatrixA[bi].rowpointer[bj+1]-subrowmatrixA[bi].rowpointer[bj];
                    }
                }
            //    ellblkwidth[ellblk]=bwidth;
            //    ellblk++;
        
                for (int ri = 0; ri < rowlength; ri++)
                {
                    for (int j = subrowmatrixA[bi].rowpointer[ri]; j < subrowmatrixA[bi].rowpointer[ri+1]; j++)
                    {
                        int temp = j - subrowmatrixA[bi].rowpointer[ri];
                        ell_colIdx[elloffset + temp * rowlength + ri] = subrowmatrixA[bi].columnindex[j];
                    //    Block_Val[blknnz[rowblock_ptr[blki] + bi] + temp * rowlength + ri] = subrowmatrixA[bi].value[j] ;
                        Blockell_Val[elloffset + temp * rowlength + ri] = subrowmatrixA[bi].value[j];
                    }
                }
                elloffset += bwidth * rowlength;
            }
            //HYB
            else if (Format[rowblock_ptr[blki]+bi] == 3)
            {
                int coocount=0;
                for (int ri = 0; ri < rowlength; ri++)
                {
                
                    int stop= (subrowmatrixA[bi].rowpointer[ri+1]- subrowmatrixA[bi].rowpointer[ri]) <= blkwidth[rowblock_ptr[blki]+bi] ? subrowmatrixA[bi].rowpointer[ri+1] :
                                                                                                                                        subrowmatrixA[bi].rowpointer[ri] + blkwidth[rowblock_ptr[blki]+bi] ;
                        
                    for (int j = subrowmatrixA[bi].rowpointer[ri]; j <stop; j++)
                    {
                        int temp = j - subrowmatrixA[bi].rowpointer[ri];
                        hyb_ellcolIdx[hyboffset+temp * rowlength + ri] = subrowmatrixA[bi].columnindex[j];
                    //    Block_Val[blknnz[rowblock_ptr[blki]+bi]+temp * rowlength + ri] = subrowmatrixA[bi].value[j];
                        Blockhyb_Val[hyboffset+temp * rowlength + ri] = subrowmatrixA[bi].value[j];
                  
                    }
                    for (int k=stop;k<subrowmatrixA[bi].rowpointer[ri+1];k++)
                    {
                    //    Block_Val[blknnz[rowblock_ptr[blki]+bi]+ blkwidth[rowblock_ptr[blki]+bi] * rowlength + coocount] = subrowmatrixA[bi].value[k] ;
                        Blockhyb_Val[hyboffset+ blkwidth[rowblock_ptr[blki]+bi] * rowlength + coocount] = subrowmatrixA[bi].value[k];
                      
                        hyb_ellcolIdx[hyboffset+ blkwidth[rowblock_ptr[blki]+bi] * rowlength + coocount]= subrowmatrixA[bi].columnindex[k];
                        hyb_coorowIdx[hybcoonnzsum+coocount]=ri;
                        coocount++;  
                    }
                }
                hybcoonnzsum += blknnz[rowblock_ptr[blki] +bi +1]- blknnz[rowblock_ptr[blki] +bi] - blkwidth[rowblock_ptr[blki] +bi] * rowlength ;   //blkhybcoonum[rowblock_ptr[blki]+bi] ;
                hyboffset += blkwidth[rowblock_ptr[blki]+bi] * rowlength + coocount;
            }
        //dense
            else if (Format[rowblock_ptr[blki]+bi] == 4)
            {
                for (int ri = 0; ri < rowlength; ri++)
                {
                    for (int j = subrowmatrixA[bi].rowpointer[ri]; j < subrowmatrixA[bi].rowpointer[ri+1]; j++)
                    {
                        Blockdense_Val[denseoffset + ri * collength + subrowmatrixA[bi].columnindex[j]]= subrowmatrixA[bi].value[j];
                    //    Block_Val[blknnz[rowblock_ptr[blki]+bi] + ri * collength + subrowmatrixA[bi].columnindex[j]]= subrowmatrixA[bi].value[j];
                    }
                }
                denseoffset += rowlength *collength;

            }
        //dense row
            else if (Format[rowblock_ptr[blki]+bi] == 5)
            {
                for (int ri = 0; ri < rowlength; ri++)
                {
                    if (subrowmatrixA[bi].rowpointer[ri+1] - subrowmatrixA[bi].rowpointer[ri] == collength)
                    {
                        denserowid[densrow_rid]=ri;
                        densrow_rid ++;
                        for (int j = subrowmatrixA[bi].rowpointer[ri]; j < subrowmatrixA[bi].rowpointer[ri+1]; j++)
                        {
                        //    Block_Val[blknnz[rowblock_ptr[blki]+bi]+denserow_vid] = subrowmatrixA[bi].value[j];
                            Blockdenserow_Val[denserow_vid]= subrowmatrixA[bi].value[j];
                            denserow_vid ++;
                        }
                    }
                }
            //    denserowptr[denrowcount+1] = densrow_rid ;
            //    denrowcount ++ ;
            }
            //dense column
            else if (Format[rowblock_ptr[blki]+bi] == 6)
            {
                for (int j=subrowmatrixA[bi].rowpointer[0];j < subrowmatrixA[bi].rowpointer[1];j ++)
                {
                    int ci = subrowmatrixA[bi].columnindex[j] ;
                    densecolid[densecol_cid] =ci ;
                    densecol_cid ++;
                }
            //    densecolptr[dencolcount +1] = densecol_cid ;
            //    dencolcount ++ ;
                for (int ri = 0; ri < rowlength; ri++)
                {
                    for (int j = subrowmatrixA[bi].rowpointer[ri]; j < subrowmatrixA[bi].rowpointer[ri+1]; j++)
                    {
                        int temp = j - subrowmatrixA[bi].rowpointer[ri];
                        Blockdensecol_Val[dencoloffset + temp * rowlength + ri] = subrowmatrixA[bi].value[j];
                    //    printf("value[%d] = %d\n",dencoloffset + temp * rowlength + ri,(int )Blockell_Val[dencoloffset + temp * rowlength + ri]) ;
                    }
                }
                dencoloffset += blknnz[rowblock_ptr[blki]+bi + 1] - blknnz[rowblock_ptr[blki]+bi] ;
             //   printf("dencoloffset = %d\n",dencoloffset) ;
            }
        
        }
           
        for (int bi=0;bi<rowbnum;bi++)
        {
            free(subrowmatrixA[bi].value);
            free(subrowmatrixA[bi].columnindex);
            free(subrowmatrixA[bi].rowpointer);
        }
        free(subrowmatrixA);
        free(num);
       
    }

    gettimeofday(&t2, NULL);
    double time_transstep2  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("transform_step2 runtime    = %4.5f sec\n", time_transstep2/1000.0);



    //MultiCOO
    MAT_VAL_TYPE *multicoo_Val=(MAT_VAL_TYPE*)malloc((coosize)*sizeof(MAT_VAL_TYPE));
	int  *multicoo_Col=(int*)malloc((coosize)*sizeof(int));
	int *multicoo_Ptr=(int *)malloc(((BLOCK_SIZE +1) ) * rbnum *sizeof(int));
    memset(multicoo_Ptr,0,((BLOCK_SIZE +1) ) * rbnum *sizeof(int)) ;

    

    int multicoo_offset =0 ;
   
 /*   for (int blki =0;blki <rbnum ;blki ++)
    {
        int multicoo_vid =0;
        
        int rowlen = blki == rbnum -1 ? rowA - (rbnum-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
        int rowbnum = rowblock_ptr[blki +1] - rowblock_ptr[blki];

        for (int ri =0;ri < rowlen ; ri ++)
        {
            for (int bi =0;bi <rowbnum ;bi ++)
            {
            //   int collen = columnid[rowblock_ptr[blki] + bi] == cbnum -1 ? colA - (cbnum-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
                if (Format[rowblock_ptr[blki] + bi] ==1)  //if COO
                {
                    //   while (nnzid <blknnz[rowblock_ptr[blki] + bi +1]- blknnz[rowblock_ptr[blki] + bi])
                        for (int nnzid =0;nnzid < blknnz[rowblock_ptr[blki] + bi +1]- blknnz[rowblock_ptr[blki] + bi]; nnzid ++)
                        {
                            int row = coo_rowIdx[multicoo_offset + nnzid] ;
                            int col = coo_colIdx[multicoo_offset + nnzid] ;
                            int val = Blockcoo_Val[multicoo_offset + nnzid] ;
                            if (row == ri) 
                            {
                                multicoo_Val[multicoonnz[blki]+ multicoo_vid] = val;
                                multicoo_Col[smulticoonnz[blki]+ multicoo_vid] = col + columnid[rowblock_ptr[blki] + bi] * BLOCK_SIZE ;
                                multicoo_vid  ++ ;
                            }
                        }  
                    multicoo_offset +=  blknnz[rowblock_ptr[blki] + bi +1]- blknnz[rowblock_ptr[blki] + bi] ;
                }
            
            }
            multicoo_Ptr[blki * BLOCK_SIZE + ri +1] = multicoo_vid ;
        }
       
    }

    for (int i=0;i<rbnum ;i++)
    {
        for (int j=0;j<BLOCK_SIZE +1 ;j++)
        {
             printf("%i    ",multicoo_Ptr[i * BLOCK_SIZE + j]);
        }
        printf("\n");
         printf("\n");
    }
    printf("\n");

*/
   




 /*   for (int i=0;i< hybellsize+hybcoosize ; i++)
    {
        if (Blockhyb_Val[i] < 0)
        {
            printf("errval= %f,colidx = %d\n",Blockhyb_Val[i],hyb_ellcolIdx[i]);
        }
    }
*/

/*    for(int i =0;i< rowblock_ptr[rbnum]+1 ;i++)
    {
        printf("%d   ",denserowptr[i]);
    }
    printf("\n");
    printf("\n");

    for(int i =0;i< denserowptr[rowblock_ptr[rbnum]] ;i++)
    {
        printf("%d   ",denserowid[i]);
    }
    printf("\n");
     printf("\n");
    for(int i =0;i< denserow_size ;i++)
    {
        printf("%d   ",(int)Blockdenserow_Val[i]);
    }
    printf("\n");
*/

/*for (int i=0;i<blknnz[nnzbl];i++)
    {
        printf("%f   ",Block_Val[i]);
    }

 printf("csr\n");
    for (int i=0;i<csrsize;i++)
    {
        printf("%f   ",Blockcsr_Val[i]);
    }
    printf("\n");
    printf("\n");
    printf("coo\n");
    for (int i=0;i<coosize;i++)
    {
        printf("%f   ",Blockcoo_Val[i]);
    }
    printf("\n");
    printf("\n");
     printf("hyb\n");
    for (int i=0;i<hybellsize+hybcoosize;i++)
    {
        printf("%f   ",Blockhyb_Val[i]);
    }
    printf("\n");
    printf("\n");
*/
/*    printf("HYB_ell width: \n");
    for (int i=0;i<rowblock_ptr[rbnum];i++)
    {
        printf("%d 's width = %d,coonum=%d \n  ",i,blkwidth[i],blkhybcoonum[i]);
    }
    printf("\n");
     printf("\n");
*/
//    printf("hybellsize+hybcoosize=%d\n",hybellsize+hybcoosize) ;
/*
    for (int i=0;i<hybellsize+hybcoosize;i++)
    {
    //    if (Blockhyb_Val[i]!=0)
 //       printf("%f   ,",Blockhyb_Val[i]);
       
        printf("%d   ,",hyb_ellcolIdx[i]);
       
 //      printf("\n");
    }
    printf("\n");
    printf("\n");
    printf("hyb_coorowIdx=%d\n",hybcoosize) ;
    for (int i=0;i<hybcoosize;i++)
    {
        printf("%d   ,",hyb_coorowIdx[i]);
    }
    printf("\n");
     printf("\n");
*/
 //   exclusive_scan(nnzb_A,nnzbl+1);

//CSR compressed Colidx
int csr_csize = csrsize % 2 ==0 ? csrsize /2 : csrsize /2 +1 ;
unsigned char *csr_compressedIdx=(char*)malloc((csr_csize)*sizeof(char));

encode(Blockcsr_Col , csr_compressedIdx,csrsize ,0);

free(Blockcsr_Col);

//COO  compressed cooIdx
   unsigned char *coo_Idx=(char*)malloc((coosize)*sizeof(char));

    int count = 0;
    for (int i = 0; i < coosize; i++)
    {
        coo_Idx[count] = (coo_rowIdx[i] << 4) + coo_colIdx[i];
        count++;
    }
  //  free(coo_colIdx);
  //  free(coo_rowIdx);

//ELl compressed cooIdx
int ell_csize = ellsize % 2 ==0 ? ellsize /2 : ellsize /2 +1 ;
unsigned char *ell_compressedIdx=(unsigned char*)malloc((ell_csize)*sizeof(unsigned char));

encode(ell_colIdx , ell_compressedIdx ,ellsize, 0 );

free(ell_colIdx);



//HYB compressed Idx
    int hyb_size = hybellsize%2==0 ? hybellsize/2 : (hybellsize/2)+1 ;
    unsigned char *hybIdx=(unsigned char*)malloc((hyb_size+hybcoosize)*sizeof(unsigned char));

    int blkoffset=0;
    hybcoonnzsum=0;
    int hc_offset=0;
    for(int blki=0;blki<rbnum;blki++)
    {
        int rowlength= blki==rbnum-1 ? rowA-(rbnum-1)* BLOCK_SIZE : BLOCK_SIZE ;
        for (int blkj=rowblock_ptr[blki];blkj<rowblock_ptr[blki+1];blkj++)
        {
            if (Format[blkj]==3)
            {
                encode(hyb_ellcolIdx + blkoffset , hybIdx, blkwidth[blkj]*rowlength,hc_offset);
            
                hc_offset += (rowlength * blkwidth[blkj])  % 2 ==0 ? (rowlength * blkwidth[blkj]) / 2 : (rowlength * blkwidth[blkj] / 2 )+ 1 ;
            //    printf ("%d , %d\n",blkj,hc_offset);
                for (int i = 0; i < blknnz[blkj+1]- blknnz[blkj] - blkwidth[blkj] * rowlength; i++)
                {
                    hybIdx[hc_offset+i] = (hyb_coorowIdx[hybcoonnzsum+i] << 4) + hyb_ellcolIdx[blkoffset+ rowlength * blkwidth[blkj]  + i];
                }
                hybcoonnzsum += blknnz[blkj+1]- blknnz[blkj] - blkwidth[blkj] * rowlength;
                
                blkoffset += blkwidth[blkj] * rowlength + blknnz[blkj+1]- blknnz[blkj] - blkwidth[blkj] * rowlength;
            
                hc_offset += blknnz[blkj+1]- blknnz[blkj] - blkwidth[blkj] * rowlength ;

            }
        }
    }
    free(hyb_ellcolIdx);
    free(hyb_coorowIdx);

    printf("Format transform success\n");



    INDEX_DATA_TYPE num_f = 240;
    INDEX_DATA_TYPE num_b = 15;

	MAT_VAL_TYPE *x = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * colA);
	for (int i = 0; i < colA; i++)
	{
		x[i] = i % 10;
	}
	MAT_VAL_TYPE *y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
	for (int i = 0; i < rowA; i++)
	{
		MAT_VAL_TYPE sum = 0;
		for (int j = matrixA.rowpointer[i]; j < matrixA.rowpointer[i+1]; j++)
		{
			sum += matrixA.value[j] * x[matrixA.columnindex[j]];
		}
		y_golden[i] = sum;
	}

    // spmv using block csr
	MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);


    int ellwoffset=0;
    int hybwoffset=0;
    int hybidxoffset=0;
    denseoffset =0;
    int coooffset= 0 ;
    int csroffset = 0;
    int csrcount =0;
//    dencolcount =0;
//    denrowcount =0;
int dnsrowoffset =0;
int dnscoloffset =0;


 gettimeofday(&t1, NULL);

//multicoo_offset =0;

    for (int blki = 0; blki < rbnum; blki++)
    {
        int rowlength=  blki== rbnum-1 ? rowA-(rbnum-1)*BLOCK_SIZE : BLOCK_SIZE ;
       
        // clear y covered by the block row
	//	int blocksize;
	//	blocksize= blki == (rbnum-1) ? 
        for (int ri = 0; ri < rowlength; ri++)
        {
            y[blki * BLOCK_SIZE + ri] = 0;
        }



        // for each block in the block row
        for (int blkj = rowblock_ptr[blki]; blkj < rowblock_ptr[blki+1]; blkj++)
        {
			int collength = columnid[blkj] == cbnum-1 ? colA - (cbnum-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
			int x_offset = columnid[blkj] * BLOCK_SIZE;


            //CSR
            if (Format[blkj]==0)
            {
                 // for each row in the block
                for (int ri = 0; ri < rowlength; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    // for each nonzero in the row of the block
                    // the last row uses nnzlocal
                    int stop = ri == rowlength - 1 ? (blknnz[blkj+1]-blknnz[blkj]) : Blockcsr_Ptr[ri+1+csrcount];
                    for (int rj = Blockcsr_Ptr[csrcount  +ri]; rj < stop; rj++)
                    {
                        int csrcol = (csroffset + rj) % 2 ==0 ? (csr_compressedIdx[(csroffset + rj) / 2] & num_f )>> 4 : 
                                                                csr_compressedIdx[(csroffset + rj) / 2 ] & num_b ;
                        sum += x[x_offset + csrcol] * Blockcsr_Val[csroffset+rj];
                    }
                    y[blki * BLOCK_SIZE + ri] += sum;
                }
                csroffset += blknnz[blkj+1]-blknnz[blkj] ;
                csrcount += rowlength ;
            }
            //COO
            else if (Format[blkj]==1 )
            {
                for (int bnnzid= 0; bnnzid < blknnz[blkj+1]-blknnz[blkj]; bnnzid++)
                {
                    int row= (coo_Idx[coooffset+bnnzid] & num_f) >> 4;//coo_rowIdx[coooffset+bnnzid];//
                    int col= coo_Idx[coooffset+bnnzid] & num_b;//coo_colIdx[coooffset+bnnzid] ; //
                //    printf("row=%i,col= %i,val = %f,x=%f\n",row,col,Blockcoo_Val[coooffset+bnnzid],x[x_offset + col]);
                    y[blki * BLOCK_SIZE+ row] += Blockcoo_Val[coooffset+bnnzid] * x[x_offset + col]; 
                //    printf("y[%i ]= %f\n",blki * BLOCK_SIZE+ row,y[blki * BLOCK_SIZE+ row]);
                 MAT_VAL_TYPE sum = Blockcoo_Val[coooffset+bnnzid] * x[x_offset + col];  
            
                }
                coooffset += blknnz[blkj+1]-blknnz[blkj] ;
             //   printf("coooffset = %i\n",coooffset);
            }

        
            //ELL
            else if (Format[blkj]==2 )
            {
            //    for each row in the block
                for (int ri = 0; ri < rowlength; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    // for each nonzero in the row of the block
                    // the last row uses nnzlocal
                        for (int j = 0; j < blkwidth[blkj]; j++)
                        {
                            int ellcol = (ellwoffset+ j * rowlength + ri) % 2 ==0 ? (ell_compressedIdx[(ellwoffset+ j * rowlength + ri) / 2] & num_f )>> 4 : 
                                                                                     ell_compressedIdx[(ellwoffset+ j * rowlength + ri) / 2 ] & num_b ;
                            if (Blockell_Val[ellwoffset + j * rowlength + ri ]!=0 )
                            {
                                sum += Blockell_Val[ellwoffset+j * rowlength + ri] * x[x_offset+ ellcol];  
                            }
                        }
                        y[blki * BLOCK_SIZE + ri] += sum;
                      
                }
            
                ellwoffset+=blkwidth[blkj]*rowlength;
            }
            //HYB
            else if (Format[blkj]==3 )
            {
                for (int ri = 0; ri < rowlength; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    for (int j = 0; j < blkwidth[blkj]; j++)
                    {
                    //    if (Blockhyb_Val[hybwoffset + j * rowlength + ri] != 0)
                    //    {
                            int hybcol = ( j * rowlength + ri)%2 == 0 ?  (hybIdx[hybidxoffset+(j * rowlength + ri)/2] & num_f )>> 4 : 
                                                                          hybIdx[hybidxoffset+(j * rowlength + ri)/2]  & num_b ;

                        //   printf("hybcol=%i,val = %f,x=%f\n",hybcol, Blockhyb_Val[hybwoffset + j * rowlength + ri],x[x_offset + hybcol]);
                            sum+= Blockhyb_Val[hybwoffset + j * rowlength + ri] * x[x_offset + hybcol];  
                     //   }
                    }
                    y[blki * BLOCK_SIZE + ri] += sum;
                    // printf("eformat=%i,sum= %f\n",Format[blkj],sum);
                }
                int offset = hybwoffset + rowlength * blkwidth[blkj];
                hybidxoffset += (rowlength * blkwidth[blkj]) % 2 ==0 ? rowlength * blkwidth[blkj] / 2 : (rowlength * blkwidth[blkj] / 2 )+ 1 ;
                for (int i = 0; i < blknnz[blkj+1]- blknnz[blkj] - blkwidth[blkj] * rowlength; i++)
                {
                    int rowidx=(hybIdx[hybidxoffset + i] & num_f) >> 4;
                    int colidx= hybIdx[hybidxoffset + i] &  num_b;
                //    if (rowidx <0)
                //        printf("rowidx=%d\n",rowidx);
                    y[blki * BLOCK_SIZE +rowidx] += Blockhyb_Val[offset + i] * x[x_offset + colidx]; 
                  

                }
            
                hybwoffset+=blkwidth[blkj] * rowlength + blknnz[blkj+1]- blknnz[blkj] - blkwidth[blkj] * rowlength;
                hybidxoffset += blknnz[blkj+1]- blknnz[blkj] - blkwidth[blkj] * rowlength ;
            }
            //dense
           else if (Format[blkj] == 4)
           {
                for (int ri = 0; ri < BLOCK_SIZE; ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    // for each nonzero in the row of the block
                    // the last row uses nnzlocal
                //	int stop = ri == BLOCK_SIZE - 1 ? (nnzb_A[blkj+1]-nnzb_A[blkj]) : BlockA_Ptr[ri+1+blkj*BLOCK_SIZE];
                    for (int rj = ri * collength; rj < (ri +1)*collength; rj++)
                    {
                        int densecol=rj % collength ;
                        sum += x[x_offset + densecol] * Blockdense_Val[denseoffset +rj];
                    }
                    y[blki * BLOCK_SIZE + ri] += sum;
                   
                }

                denseoffset += rowlength * collength ;
           }
       //dense row
            else if (Format[blkj] == 5)
            {
                for (int ri=denserowptr[blkj]; ri < denserowptr[ blkj +1 ];ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    for (int rj = 0; rj < collength; rj++)
                    {
                    //    int denserowcol=rj;
                        sum += x[x_offset + rj] * Blockdenserow_Val[dnsrowoffset + (ri - denserowptr[blkj]) * collength +rj];
                    }
                    y[blki * BLOCK_SIZE + denserowid[ri]] += sum;
                   
                }
                dnsrowoffset +=  blknnz[blkj+1]-blknnz[blkj] ;
            }
            //dense column
            else if (Format[blkj] == 6)
            {
                for (int ri=0 ; ri < rowlength;ri++)
                {
                    MAT_VAL_TYPE sum = 0;
                    for (int rj = densecolptr[blkj]; rj < densecolptr[blkj +1]; rj++)
                    {
                        sum += Blockdensecol_Val[dnscoloffset + (rj-densecolptr[blkj]) * rowlength + ri] * x[x_offset+ densecolid[rj]];  
                    }
                    y[blki * BLOCK_SIZE + ri] += sum;
                  
                }
                dnscoloffset +=  blknnz[blkj+1]-blknnz[blkj] ;
            }
        
        }

    /*    //multicoo

        for (int ri = 0; ri < rowlength; ri++)
        {
            MAT_VAL_TYPE sum = 0;
            // for each nonzero in the row of the block
            // the last row uses nnzlocal
        //    int stop = ri == rowlength - 1 ? (blknnz[blkj+1]-blknnz[blkj]) : Blockcsr_Ptr[ri+1+csrcount];
            int stop = multicoo_Ptr[blki * BLOCK_SIZE + ri +1] ;
            for (int rj = multicoo_Ptr[blki * BLOCK_SIZE + ri ]; rj < stop; rj++)
            {
            //    int csrcol = (csroffset + rj) % 2 ==0 ? (csr_compressedIdx[(csroffset + rj) / 2] & num_f )>> 4 : 
            //                                            csr_compressedIdx[(csroffset + rj) / 2 ] & num_b ;
                int col = multicoo_Col[multicoo_Ptr[blki * BLOCK_SIZE ] + rj];

                sum += x[col] * multicoo_Val[multicoo_Ptr[blki * BLOCK_SIZE ] + rj];
            }
            y[blki * BLOCK_SIZE + ri] += sum;
        }
    */


    }

    gettimeofday(&t2, NULL);
    double cputime  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CPU runtime    = %4.5f sec\n", cputime/1000.0);

    // check results
	int errcount = 0;
	for (int i = 0; i < rowA; i++)
	{
		if (y[i] != y_golden[i])
		{

			errcount++;
		//	printf("%f    %f,%d\n",y[i],y_golden[i],i);

		}
		   
	}
	printf("spmv errcount = %i\n", errcount);



//    for(int i=0;i<rowblock_ptr[rbnum];i++)
//    {
//        printf("i=%i,nnz = %i,format = %i\n",i,blknnz[i+1]-blknnz[i],Format[i] ) ;
//    }

 //CSR
    free(Blockcsr_Val);
    free(csr_compressedIdx);
    free(Blockcsr_Ptr);
//COO
    free(Blockcoo_Val);
    free(coo_Idx);
//ELL
    free(Blockell_Val);
    free(ell_compressedIdx);
//HYB
    free(Blockhyb_Val);
    free(hybIdx);

//dense
    free(Blockdense_Val);

//denserow
    free(Blockdenserow_Val);
    free(denserowid);
    free(denserowptr);

//densecol
    free(Blockdensecol_Val);
    free(densecolid);
    free(densecolptr);


    free(matrixA.value);
    free(matrixA.columnindex);
    free(matrixA.rowpointer);

}