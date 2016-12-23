#!/bin/bash
# Copyright (c) 2016-2017, Intel Corporation
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
##################################################################################
# Alexander Heinecke (Intel Corp.)
##################################################################################

echo "Please use sufficient affinities when running this benchmark"
echo "e.g.:"
echo "export OMP_NUM_THREADS=X"
echo "export KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=67
export KMP_AFFINITY=granularity=fine,compact,1,0

numactl --membind=1 ./pyfr_gemm_rm 150 2048 125 1000
numactl --membind=1 ./pyfr_gemm_rm 150 48000 125 1000
numactl --membind=1 ./pyfr_gemm_rm 150 96000 125 1000

numactl --membind=1 ./pyfr_gemm_cm 150 2048 125 1000
numactl --membind=1 ./pyfr_gemm_cm 150 48000 125 1000
numactl --membind=1 ./pyfr_gemm_cm 150 96000 125 1000

numactl --membind=1 ./pyfr_gemm_rm 105 2048 75 1000
numactl --membind=1 ./pyfr_gemm_rm 105 48000 75 1000
numactl --membind=1 ./pyfr_gemm_rm 105 96000 75 1000

numactl --membind=1 ./pyfr_gemm_cm 105 2048 75 1000
numactl --membind=1 ./pyfr_gemm_cm 105 48000 75 1000
numactl --membind=1 ./pyfr_gemm_cm 105 96000 75 1000

numactl --membind=1 ./pyfr_driver_asp_reg ./mats/p3/hex/m6-sp.mtx 48000 10000
