#!/bin/sh
#
# Copyright (c) 2015, Intel Corporation
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

M="4 10 20 35 56 84"
M="4 12 20 36 56 84"
#M="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100"
#M="8 16 24 40 56 88"
#M="56"

cd ./../../

make realclean
make generator_backend

#exit if compiler fails
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

cd ./samples/generator/

# select architecture for validation
ARCH=hsw
PREC=DP
if [ $# -eq 2 ]
then
  ARCH=$1
  PREC=$2
fi
ASM=1

# select precision 
if [ "${PREC}" == 'DP' ]; then
  DATATYPE="double"
elif [ "${PREC}" == 'SP' ]; then    
  DATATYPE="float"
fi

N="9"
for m in ${M}
do
  for n in ${N}
  do
    K="${m} ${N}"
    #K="${m}"
    for k in ${K}
    do 
      rm -rf kernel_${m}_${n}_${k}_${PREC}.*
      rm -rf xgemm_${m}_${n}_${k}_${PREC}
      ./../../bin/generator dense kernel_${m}_${n}_${k}_${PREC}.h dense_test_mul $m $n $k $m $k $m 1 1 1 1 ${ARCH} nopf ${PREC}
      if [ $ASM -eq 1 ]
      then
        ./../../bin/generator dense_asm kernel_${m}_${n}_${k}_${PREC}.s dense_test_mul $m $n $k $m $k $m 1 1 1 1 ${ARCH} nopf ${PREC}
      fi
      if [ "${ARCH}" == 'wsm' ]; then
        #icpc -O3 -msse3 -ansi-alias -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DREALTYPE=${DATATYPE} -static-intel validation.cpp -o xgemm_${m}_${n}_${k}_${PREC}
        #./xgemm_${m}_${n}_${k}_${PREC}
        echo "wsm is currently supported by this version of the generator"
      elif [ "${ARCH}" == 'snb' ]; then
        icpc -O3 -mavx -ansi-alias -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" -static-intel validation.cpp -o xgemm_${m}_${n}_${k}_${PREC}
        ./xgemm_${m}_${n}_${k}_${PREC}
        if [ $ASM -eq 1 ]
        then
          as kernel_${m}_${n}_${k}_${PREC}.s -o kernel_${m}_${n}_${k}_${PREC}.o
          icpc -O2 -mavx-DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DREALTYPE=${DATATYPE} -DUSE_ASM_DIRECT -static-intel validation.cpp kernel_${m}_${n}_${k}_${PREC}.o -o xgemm_${m}_${n}_${k}_${PREC}_asm
          ./xgemm_${m}_${n}_${k}_${PREC}_asm
        fi
      elif [ "${ARCH}" == 'hsw' ]; then
        #icpc -O2 -xCORE_AVX2 -fma -D__USE_MKL -mkl -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" -static-intel validation.cpp -o xgemm_${m}_${n}_${k}_${PREC}
        icpc -O2 -xCORE_AVX2 -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DREALTYPE=${DATATYPE} -DGEMM_HEADER=\"kernel_${m}_${n}_${k}_${PREC}.h\" -static-intel validation.cpp -o xgemm_${m}_${n}_${k}_${PREC}
        ./xgemm_${m}_${n}_${k}_${PREC}
        if [ $ASM -eq 1 ]
        then
          as kernel_${m}_${n}_${k}_${PREC}.s -o kernel_${m}_${n}_${k}_${PREC}.o
          icpc -O2 -mavx -D__AVX2__ -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DREALTYPE=${DATATYPE} -DUSE_ASM_DIRECT -static-intel validation.cpp kernel_${m}_${n}_${k}_${PREC}.o -o xgemm_${m}_${n}_${k}_${PREC}_asm
          ./xgemm_${m}_${n}_${k}_${PREC}_asm
        fi
      elif [ "${ARCH}" == 'knc' ]; then
        #icpc -O3 -mmic -fma -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DREALTYPE=${DATATYPE} -static-intel validation.cpp -o xgemm_${m}_${n}_${k}
        #scp xgemm_${m}_${n}_${k} mic0:
	#ssh mic0 "./xgemm_${m}_${n}_${k}"
        echo "knc is currently NOT supported by this version of the generator!"
      elif [ "${ARCH}" == 'knl' ]; then
         #icpc -O3 -xMIC-AVX512 -fma -ansi-alias -static-intel -DNDEBUG -DMY_M=$m -DMY_N=$n -DMY_K=$k -DREALTYPE=${DATATYPE} validation.cpp -o xgemm_${m}_${n}_${k}
         #/nfs_home/aheineck/Simulators/sde/sde-buildkit-internal-rs-7.23.0-2015-04-30-lin/sde64 -knl -mix -- ./xgemm_${m}_${n}_${k}
         #./xgemm_${m}_${n}_${k}
        echo "knl is currently NOT supported by this version of the generator!"
      else
        echo "unsupported architecture!"
      fi
    done
  done
done
