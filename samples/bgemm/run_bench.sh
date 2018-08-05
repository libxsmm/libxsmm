#!/bin/bash
#############################################################################
# Copyright (c) 2017-2018, Intel Corporation                                #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################

if [ $1 = "-h" ]
then
  echo "Usage: $(basename $0) matrices.txt iters numa (1-mcdram/0-DDR)"
  exit
fi

fn=${1:-"deepbench_opt.txt"}
ITERS=${2:-100}
NUMA=${3:-1}

NUMACTL="${TOOL_COMMAND}"
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --preferred=${NUMA} ${TOOL_COMMAND}"
  fi
fi

if [[ -z "${OMP_NUM_THREADS}" ]]; then
  echo "using defaults for OMP settings!"
  export KMP_HW_SUBSET=1T
  export KMP_AFFINITY=compact,granularity=fine
  export KMP_AFFINITY=proclist=[1-67],granularity=thread,explicit,norespect
  export OMP_NUM_THREADS=67
else
  echo "using environment OMP settings!"
fi

#./bgemm iters M N K order BM BN BK B_M B_N B_K K_unroll
# current bgemm only supports non-transpose GEMM, TODO: transpose support
_bin="$NUMACTL ./bgemm"
_it=$ITERS

function run_bsgemm {
_M=$1
_N=$2
_K=$3
_AT=$4
_BT=$5
_mb=32
_nb=32
_kb=32
_order=0
_mb1=1
_nb1=1
_kb1=1
_kb2=1

if [[ $# -gt 5 ]]
then
  _mb=$6
  _nb=$7
  _kb=$8
fi

if [[ $# -gt 8 ]]
then
  _order=$9
  _mb1=${10}
  _nb1=${11}
  _kb1=${12}
  _kb2=${13}
  st=${14}
  _it=$((st*ITERS))
fi

if [[ "$_mb" -gt "_M" ]]
then
  _mb=$_M
fi
if [[ "$_nb" -gt "_N" ]]
then
  _nb=$_N
fi
if [[ "$_kb" -gt "_K" ]]
then
  _kb=$_K
fi

if [[ "$((_M % _mb))" -gt 0 ]]
then
  #_mb=$_M
  _M=$((_mb*(_M/_mb+1)))
fi
if [[ "$((_N % _nb))" -gt 0 ]]
then
  #_nb=$_N
  _N=$((_nb*(_N/_nb+1)))
fi
if [[ "$((_K % _kb))" -gt 0 ]]
then
  #_kb=$_K
  _K=$((_kb*(_K/_kb+1)))
fi

if [[ "$_AT" == "T" ]]
then
  echo "!!! $_M $_N $_K $_AT $_BT - Not supported !!!, doing $_N $_M $_K N N instead"
  t_M=$_M
  t_mb=$_mb
  _M=$_N
  _N=$t_M
  _mb=$_nb
  _nb=$t_mb
fi
if [[ "$_BT" == "T" ]]
then
  echo "!!! $_M $_N $_K $_AT $_BT - Not supported !!!, doing $_M $_K $_N N N instead"
  t_K=$_K
  t_kb=$_kb
  _K=$_N
  _N=$t_K
  _kb=$_nb
  _nb=$t_kb
fi

echo "$_bin $_M $_N $_K $_mb $_nb $_kb $_order $_it $_mb1 $_nb1 $_kb1 $_kb2"
$_bin $_M $_N $_K $_mb $_nb $_kb $_order $_it $_mb1 $_nb1 $_kb1 $_kb2
echo "--------------------------------------------------------------------------------------"
}

nc=$(wc -l $fn)
idx=1

cat $fn | while read line
do
  if [ ! -z "$line" ]; then
    echo -n "($idx/$nc)  "
    run_bsgemm $line
  fi
  idx=$((idx+1))
done
