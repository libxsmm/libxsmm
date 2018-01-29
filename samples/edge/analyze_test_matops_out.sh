#!/bin/bash
#
# Copyright (c) 2018, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

if [ $# -eq 2 ]
then
  OUTFILE=$1
  FORMAT=$2
else
  echo "you have to provide an outfile (stdout of test_matops.sh) and csc/csr!"
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
