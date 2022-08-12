input="2757-matrix.csv"

{
  read
  i=1
  while IFS=',' read -r mid Group Name rows cols nonzeros
  do
    echo "$mid $Group $Name $rows $cols $nonzeros"
    #./trsv_test ~/matrix-lu176/$name.l
    #./sptrsv_d0 /home/weifeng/UFget/MM/$group/$name/$name.mtx
    ./spmv_d1 /home/weifeng/UFget/MM/$Group/$Name/$Name.mtx
    i=`expr $i + 1`
  done 
} < "$input"
