#!/bin/sh
#
# Copyright (c) 2015-2016, Intel Corporation
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
MAKE=${MAKE:-make}

M="4 10 20 35 56 84"
M="4 12 20 36 56 84"
#M="4 8 12 16 20 24 27 28 29 30 31 36 40 48 56 84"
#M="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100"
#M="8 16 24 40 56 88"
#M="56"

cd ./../../

#${MAKE} realclean
${MAKE} generator

#exit if compiler fails
rc=$?; if [ $rc != 0 ]; then exit $rc; fi

cd ./samples/generator/

# select architecture for validation
ARCH=hsw
PREC=DP
if [ $# -eq 2 ]
then
  ARCH=$1
  PREC=$2
fi

# build also assembly variant
ASM=1

# set SDE to run AVX512 code
SDE_KNL="sde64 -knl -mix -- "

# select precision 
if [ "${PREC}" == 'DP' ]; then
  DATATYPE="double"
elif [ "${PREC}" == 'SP' ]; then    
  DATATYPE="float"
fi

N="9"
#N="1 2 3 4 5 6 7 8 9 10 30 31 60 62"
for m in ${M}
do
  for n in ${N}
  do
    K="${m} ${N}"
#    K="${m}"
#    K="1 2 3 4 5 6 7 8 9 10 30 31 60 62"
    for k in ${K}
    do
      lda=$m
      ldb=$k
      ldc=$m 
      rm -rf kernel_${m}_${n}_${k}_${PREC}.*
      rm -rf xgemm_${m}_${n}_${k}_${PREC}
      ./../../bin/libxsmm_gemm_generator dense kernel_${m}_${n}_${k}_${PREC}.h dense_test_mul $m $n $k $lda $ldb $ldc 1 1 1 1 ${ARCH} nopf ${PREC}
      if [ $ASM -eq 1 ]
      then
        ./../../bin/libxsmm_gemm_generator dense_asm kernel_${m}_${n}_${k}_${PREC}.s dense_test_mul $m $n $k $lda $ldb $ldc 1 1 1 1 ${ARCH} nopf ${PREC}
      fi
      if [ "${ARCH}" == 'wsm' ]; then
        icc -O3 -msse3 -ansi-alias -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" validation.c -o xgemm_${m}_${n}_${k}_${PREC}
        ./xgemm_${m}_${n}_${k}_${PREC}
        if [ $ASM -eq 1 ]
        then
          as kernel_${m}_${n}_${k}_${PREC}.s -o kernel_${m}_${n}_${k}_${PREC}.o
          icc -O2 -msse3 -ansi-alias -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DUSE_ASM_DIRECT validation.c kernel_${m}_${n}_${k}_${PREC}.o -o xgemm_${m}_${n}_${k}_${PREC}_asm
          ./xgemm_${m}_${n}_${k}_${PREC}_asm
        fi
      elif [ "${ARCH}" == 'snb' ]; then
        icc -O3 -mavx -ansi-alias -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" validation.c -o xgemm_${m}_${n}_${k}_${PREC}
        ./xgemm_${m}_${n}_${k}_${PREC}
        if [ $ASM -eq 1 ]
        then
          as kernel_${m}_${n}_${k}_${PREC}.s -o kernel_${m}_${n}_${k}_${PREC}.o
          icc -O2 -ansi-alias -mavx -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DUSE_ASM_DIRECT validation.c kernel_${m}_${n}_${k}_${PREC}.o -o xgemm_${m}_${n}_${k}_${PREC}_asm
          ./xgemm_${m}_${n}_${k}_${PREC}_asm
        fi
      elif [ "${ARCH}" == 'hsw' ]; then
        #icc -O3 -xCORE_AVX2 -fma -D__USE_MKL -mkl=sequential -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" validation.c -o xgemm_${m}_${n}_${k}_${PREC}
        icc -O2 -ansi-alias -xCORE_AVX2 -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" validation.c -o xgemm_${m}_${n}_${k}_${PREC}
        ./xgemm_${m}_${n}_${k}_${PREC}
        if [ $ASM -eq 1 ]
        then
          as kernel_${m}_${n}_${k}_${PREC}.s -o kernel_${m}_${n}_${k}_${PREC}.o
          icc -O2 -ansi-alias -mavx -D__AVX2__ -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DUSE_ASM_DIRECT validation.c kernel_${m}_${n}_${k}_${PREC}.o -o xgemm_${m}_${n}_${k}_${PREC}_asm
          ./xgemm_${m}_${n}_${k}_${PREC}_asm
        fi
      elif [ "${ARCH}" == 'knc' ]; then
        icc -O2 -ansi-alias -mmic -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" validation.c -o xgemm_${m}_${n}_${k}_${PREC}
        scp ./xgemm_${m}_${n}_${k}_${PREC} mic0:
        ssh mic0 "./xgemm_${m}_${n}_${k}_${PREC}"
      elif [ "${ARCH}" == 'knl' ]; then
        icc -O2 -ansi-alias -xCOMMON-AVX512 -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" validation.c -o xgemm_${m}_${n}_${k}_${PREC}
        ${SDE_KNL} ./xgemm_${m}_${n}_${k}_${PREC}
        if [ $ASM -eq 1 ]
        then
          as kernel_${m}_${n}_${k}_${PREC}.s -o kernel_${m}_${n}_${k}_${PREC}.o
          icc -O2 -ansi-alias -xCOMMON_AVX512 -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DUSE_ASM_DIRECT validation.c kernel_${m}_${n}_${k}_${PREC}.o -o xgemm_${m}_${n}_${k}_${PREC}_asm
          ${SDE_KNL} ./xgemm_${m}_${n}_${k}_${PREC}_asm
        fi
      elif [ "${ARCH}" == 'noarch' ]; then
        icc -O2 -ansi-alias -xHOST -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DMY_LDA=$lda -DMY_LDB=$ldb -DMY_LDC=$ldc -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" validation.c -o xgemm_${m}_${n}_${k}_${PREC}
        ./xgemm_${m}_${n}_${k}_${PREC}
      else
        echo "unsupported architecture!"
      fi
    done
  done
done
