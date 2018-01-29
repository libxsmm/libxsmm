#!/bin/bash

if [ $# -eq 2 ]
then
  OUTFILE=$1
  FORMAT=$2
else
  echo "you have to provide an outfile (stdout of test_matrixops.sh) and csc/csr!"
  exit -1
fi

TOTALFLOPS=0
WEIGHTAVGGFLOPS=0

for i in `cat ${OUTFILE} | grep PERFDUMP | grep ${FORMAT} | awk -F"," '{print $3 "," $7 "," $8 "," $10 "," $11}'`
do 
  FLOPS=`echo $i | awk -F"," '{print $2}'`
  TOTALFLOPS=`echo $TOTALFLOPS+$FLOPS | bc`
done 

for i in `cat ${OUTFILE} | grep PERFDUMP | grep ${FORMAT} | awk -F"," '{print $3 "," $7 "," $8 "," $10 "," $11}'`
do 
  FLOPS=`echo $i | awk -F"," '{print $2}'`
  GFLOPS=`echo $i | awk -F"," '{print $5}'`
  WEIGHT=`echo $FLOPS/$TOTALFLOPS | bc -l`
  WEIGHTGFLOPS=`echo $GFLOPS*$WEIGHT | bc -l`
  WEIGHTAVGGFLOPS=`echo $WEIGHTAVGGFLOPS+$WEIGHTGFLOPS | bc -l`
done 

echo $OUTFILE","$WEIGHTAVGGFLOPS
