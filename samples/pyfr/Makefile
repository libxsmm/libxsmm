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

AVX?=3
CC=icc
CFLAGS=-qopenmp -O2 -mkl -I./../../include/ -c
LDFLAGS=-mkl -L./../../lib -lxsmm

ifeq ($(AVX), 1)
CXXFLAGS+=-mavx
endif
ifeq ($(AVX), 2)
CXXFLAGS+=-xCORE-AVX2 -fma
endif
ifeq ($(AVX), 3)
CXXFLAGS+=-xCOMMON-AVX512 -fma
endif

default: pyfr_gemm_cm pyfr_gemm_rm pyfr_driver_asp_reg

pyfr_gemm_cm.o: pyfr_gemm_cm.c
	$(CC) $(CFLAGS) pyfr_gemm_cm.c

pyfr_gemm_rm.o: pyfr_gemm_rm.c
	$(CC) $(CFLAGS) pyfr_gemm_rm.c

pyfr_driver_asp_reg.o : pyfr_driver_asp_reg.c
	$(CC) $(CFLAGS) pyfr_driver_asp_reg.c

pyfr_gemm_cm: pyfr_gemm_cm.o
	$(CC) pyfr_gemm_cm.o $(LDFLAGS) -o pyfr_gemm_cm

pyfr_gemm_rm: pyfr_gemm_rm.o
	$(CC) pyfr_gemm_rm.o $(LDFLAGS) -o pyfr_gemm_rm

pyfr_driver_asp_reg: pyfr_driver_asp_reg.o
	$(CC) pyfr_driver_asp_reg.o $(LDFLAGS) -mkl -o pyfr_driver_asp_reg

clean:
	rm -rf *.o kernel.c
	rm -rf pyfr_gemm_cm pyfr_gemm_rm
