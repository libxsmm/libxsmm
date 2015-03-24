#! /usr/bin/env python
###############################################################################
## Copyright (c) 2013-2015, Intel Corporation                                ##
## All rights reserved.                                                      ##
##                                                                           ##
## Redistribution and use in source and binary forms, with or without        ##
## modification, are permitted provided that the following conditions        ##
## are met:                                                                  ##
## 1. Redistributions of source code must retain the above copyright         ##
##    notice, this list of conditions and the following disclaimer.          ##
## 2. Redistributions in binary form must reproduce the above copyright      ##
##    notice, this list of conditions and the following disclaimer in the    ##
##    documentation and/or other materials provided with the distribution.   ##
## 3. Neither the name of the copyright holder nor the names of its          ##
##    contributors may be used to endorse or promote products derived        ##
##    from this software without specific prior written permission.          ##
##                                                                           ##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       ##
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         ##
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     ##
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      ##
## HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    ##
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  ##
## TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    ##
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    ##
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      ##
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        ##
## SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              ##
###############################################################################
## Christopher Dahnken (Intel Corp.), Hans Pabst (Intel Corp.),
## Alfio Lazzaro (CRAY Inc.), and Gilles Fourestey (CSCS)
###############################################################################
import libxsmm_utilities
import math
import sys


def create_macros(RowMajor, AlignedStores, AlignedLoads, Alignment, listMNK, Threshold):
    print "#define LIBXSMM_ALIGNMENT " + str(Alignment)
    print "#define LIBXSMM_ALIGNED_STORES " + ["0", [str(Alignment), str(AlignedStores)][1 < AlignedStores]][0 != AlignedStores]
    print "#define LIBXSMM_ALIGNED_LOADS " + ["0", [str(Alignment), str(AlignedLoads)][1 < AlignedLoads]][0 != AlignedLoads]
    print "#define LIBXSMM_ROW_MAJOR " + ["0", "1"][0 != RowMajor]
    print "#define LIBXSMM_COL_MAJOR " + ["1", "0"][0 != RowMajor]
    print "#define LIBXSMM_MAX_MNK " + str(Threshold)
    maxMNK = int(Threshold ** (1.0 / 3.0) + 0.5)
    print "#define LIBXSMM_MAX_M " + str(libxsmm_utilities.max_mnk(listMNK, maxMNK, 0))
    print "#define LIBXSMM_MAX_N " + str(libxsmm_utilities.max_mnk(listMNK, maxMNK, 1))
    print "#define LIBXSMM_MAX_K " + str(libxsmm_utilities.max_mnk(listMNK, maxMNK, 2))
    listM = libxsmm_utilities.make_mlist(listMNK)
    listN = libxsmm_utilities.make_nlist(listMNK)
    listK = libxsmm_utilities.make_klist(listMNK)
    print "#define LIBXSMM_AVG_M " + str(int(float(sum(listM)) / len(listM) + 0.5))
    print "#define LIBXSMM_AVG_N " + str(int(float(sum(listN)) / len(listN) + 0.5))
    print "#define LIBXSMM_AVG_K " + str(int(float(sum(listK)) / len(listK) + 0.5))
    print
    print "#define LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C) { \\"
    print "  UINT libxsmm_m_ = (M), libxsmm_n_ = (N), libxsmm_k_ = (K); \\"
    if (0 != RowMajor):
        mnk = "&libxsmm_n_, &libxsmm_m_, &libxsmm_k_"
        amb = "(REAL*)(B), &libxsmm_n_, (REAL*)(A)"
        ldc = "(N)"
    else:
        mnk = "&libxsmm_m_, &libxsmm_n_, &libxsmm_k_"
        amb = "(REAL*)(A), &libxsmm_m_, (REAL*)(B)"
        ldc = "(M)"
    if (0 != AlignedStores):
        print "  UINT libxsmm_ldc_ = LIBXSMM_ALIGN_VALUE(UINT, REAL, " + ldc + ", LIBXSMM_ALIGNED_STORES); \\"
    else:
        print "  UINT libxsmm_ldc_ = " + ldc + "; \\"
    print "  REAL libxsmm_alpha_ = 1, libxsmm_beta_ = 1; \\"
    print "  char libxsmm_trans_ = 'N'; \\"
    print "  LIBXSMM_FSYMBOL(LIBXSMM_BLASPREC(, REAL, gemm))(&libxsmm_trans_, &libxsmm_trans_, \\"
    print "    " + mnk + ", \\"
    print "    &libxsmm_alpha_, " + amb + ", &libxsmm_k_, \\"
    print "    &libxsmm_beta_, (C), &libxsmm_ldc_); \\"
    print "}"
    print
    print "#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)"
    print "# define LIBXSMM_IMM(REAL, UINT, M, N, K, A, B, C) LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C)"
    print "#else"
    print "# define LIBXSMM_IMM(REAL, UINT, M, N, K, A, B, C) { \\"
    print "    UINT libxsmm_i_, libxsmm_j_, libxsmm_k_; \\"
    print "    const REAL *const libxsmm_a_ = (A), *const libxsmm_b_ = (B); \\"
    print "    REAL *const libxsmm_c_ = (C); \\"
    if (0 != AlignedStores):
        print "    LIBXSMM_ASSUME_ALIGNED(libxsmm_c_, LIBXSMM_ALIGNED_STORES); \\"
    if (0 != AlignedLoads and False): # TODO
        print "    LIBXSMM_ASSUME_ALIGNED(libxsmm_a_, LIBXSMM_ALIGNED_LOADS); \\"
        print "    LIBXSMM_ASSUME_ALIGNED(libxsmm_b_, LIBXSMM_ALIGNED_LOADS); \\"
    print "    LIBXSMM_PRAGMA_SIMD_COLLAPSE(2) \\"
    if (0 != RowMajor):
        print "    for (libxsmm_j_ = 0; libxsmm_j_ < (N); ++libxsmm_j_) { \\"
        print "      LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_MAX_M, LIBXSMM_AVG_M) \\"
        print "      for (libxsmm_i_ = 0; libxsmm_i_ < (M); ++libxsmm_i_) { \\"
        if (0 != AlignedStores):
            print "        const UINT libxsmm_index_ = libxsmm_i_ * LIBXSMM_ALIGN_VALUE(UINT, REAL, N, LIBXSMM_ALIGNED_STORES) + libxsmm_j_; \\"
        else:
            print "        const UINT libxsmm_index_ = libxsmm_i_ * (N) + libxsmm_j_; \\"
        print "        REAL libxsmm_r_ = libxsmm_c_[libxsmm_index_]; \\"
        print "        LIBXSMM_PRAGMA_SIMD_REDUCTION(+:libxsmm_r_) \\"
        print "        LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_MAX_K, LIBXSMM_AVG_K) \\"
        print "        for (libxsmm_k_ = 0; libxsmm_k_ < (K); ++libxsmm_k_) { \\"
        print "          libxsmm_r_ += libxsmm_a_[libxsmm_i_*(K)+libxsmm_k_] * libxsmm_b_[libxsmm_k_*(N)+libxsmm_j_]; \\"
        print "        } \\"
        print "        libxsmm_c_[libxsmm_index_] = libxsmm_r_; \\"
        print "      } \\"
        print "    } \\"
    else:
        print "    for (libxsmm_j_ = 0; libxsmm_j_ < (M); ++libxsmm_j_) { \\"
        print "      LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_MAX_N, LIBXSMM_AVG_N) \\"
        print "      for (libxsmm_i_ = 0; libxsmm_i_ < (N); ++libxsmm_i_) { \\"
        if (0 != AlignedStores):
            print "        const UINT libxsmm_index_ = libxsmm_i_ * LIBXSMM_ALIGN_VALUE(UINT, REAL, M, LIBXSMM_ALIGNED_STORES) + libxsmm_j_; \\"
        else:
            print "        const UINT libxsmm_index_ = libxsmm_i_ * (M) + libxsmm_j_; \\"
        print "        REAL libxsmm_r_ = libxsmm_c_[libxsmm_index_]; \\"
        print "        LIBXSMM_PRAGMA_SIMD_REDUCTION(+:libxsmm_r_) \\"
        print "        LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_MAX_K, LIBXSMM_AVG_K) \\"
        print "        for (libxsmm_k_ = 0; libxsmm_k_ < (K); ++libxsmm_k_) { \\"
        print "          libxsmm_r_ += libxsmm_a_[libxsmm_k_*(M)+libxsmm_j_] * libxsmm_b_[libxsmm_i_*(K)+libxsmm_k_]; \\"
        print "        } \\"
        print "        libxsmm_c_[libxsmm_index_] = libxsmm_r_; \\"
        print "      } \\"
        print "    } \\"
    print "  }"
    print "#endif"


def create_implementation(Real, M, N, K, RowMajor, AlignedStores, AlignedLoads):
    if (0 != RowMajor):
        Rows, Cols = N, M
        l1, l2 = "b", "a"
    else: # ColMajor
        Rows, Cols = M, N
        l1, l2 = "a", "b"
    iparts = int(math.floor(Rows / 8))
    if (0 == (Rows % 8)):
        mnparts = iparts
    else:
        mnparts = iparts + 1
    print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_" + libxsmm_utilities.make_typeflag(Real) + "mm_" + str(M) + "_" + str(N) + "_" + str(K) + "(const " + Real + "* a, const " + Real + "* b, " + Real + "* c)"
    print "{"
    print "#if defined(__MIC__) || defined(__AVX512F__)"
    if (0 != AlignedStores):
        print "  const int r = LIBXSMM_ALIGN_VALUE(int, " + Real + ", " + str(Rows) + ", LIBXSMM_ALIGNED_STORES);"
    else:
        print "  const int r = " + str(Rows) + ";"
    print "  int i = 0, k = 0;"
    for mn in range(0, 8 * mnparts, 8):
        print "  {"
        mnm = min(mn + 7, Rows - 1)
        maskval = (1 << (mnm - mn + 1)) - 1
        if (255 != maskval):
            mask_inst, mask_argv = "_MASK", ", " + str(maskval)
        else:
            mask_inst, mask_argv = "", ""
        print "    const " + Real + "* src = " + l2 + "; " + Real + "* dst = c + " + str(mn) + ";"
        print "    __m512" + libxsmm_utilities.make_typepfix(Real) + " x" + l1 + "[" + str(K) + "], x" + l2 + "[" + str(K) + "], xc = MM512_LOAD" + ["U", ""][0 != AlignedLoads] + mask_inst + "_PD(dst" + mask_argv + ", _MM_HINT_NONE);"
        print
        print "    for (k = 0; k < " + str(K) + "; ++k) {"
        print "      x" + l1 + "[k] = MM512_LOAD" + ["U", ""][0 != AlignedLoads] + mask_inst + "_PD(" + l1 + " + k * " + str(Rows) + " + " + str(mn) + mask_argv + ", _MM_HINT_NONE),"
        print "      x" + l2 + "[k] = MM512_SET1_PD(src[k]);"
        print "      xc = MM512_FMADD" + mask_inst +"_PD(xa[k], xb[k], xc" + mask_argv + ");"
        print "    }"
        print "    MM512_STORE" + ["U", ""][0 != AlignedStores] + mask_inst + "_PD(dst, xc" + mask_argv + ", _MM_HINT_NONE);"
        print
        print "    for (i = 1; i < " + str(Cols) + "; ++i) {"
        print "      src += " + str(K) + "; dst += r;"
        print "      xc = MM512_LOAD" + ["U", ""][0 != AlignedLoads] + mask_inst + "_PD(dst" + mask_argv + ", _MM_HINT_NONE);"
        print
        print "      for (k = 0; k < " + str(K) + "; ++k) {"
        print "        x" + l2 + "[k] = MM512_SET1_PD(src[k]);"
        print "        xc = MM512_FMADD" + mask_inst +"_PD(xa[k], xb[k], xc" + mask_argv + ");"
        print "      }"
        print "      MM512_STORE" + ["U", ""][0 != AlignedStores] + mask_inst + "_PD(dst, xc" + mask_argv + ", _MM_HINT_NONE);"
        print "    }"
        print "  }"
    print "#else"
    print "  LIBXSMM_IMM(" + Real + ", int, " + str(M) + ", " + str(N) + ", " + str(K) + ", a, b, c);"
    print "#endif"
    print "}"


if __name__ == '__main__':
    if (7 < len(sys.argv)):
        RowMajor = int(sys.argv[1])
        AlignedStores = int(sys.argv[2])
        AlignedLoads = int(sys.argv[3])
        Alignment = int(sys.argv[4])
        Threshold = int(sys.argv[5])

        if (1 < AlignedStores and False == libxsmm_utilities.is_pot(AlignedStores)):
            raise ValueError("Memory alignment for Store instructions must be a Power of Two (POT) number!")
        if (1 < AlignedLoads and False == libxsmm_utilities.is_pot(AlignedLoads)):
            raise ValueError("Memory alignment for Load instructions must be a Power of Two (POT) number!")
        if (0 >= Alignment):
            Alignment = [1, 64][0 != Alignment] # sanitize/fallback
        elif (False == libxsmm_utilities.is_pot(Alignment)):
            raise ValueError("Memory alignment must be a Power of Two (POT) number!")

        if (0 > Threshold):
            print "#include \"libxsmm_isa.h\""
            print "#include <libxsmm.h>"
            print
            print
            M, N, K = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
            # Note: create_implementation is not yet ready to generate the single-precision implementation
            print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_smm_" + str(M) + "_" + str(N) + "_" + str(K) + "(const float* a, const float* b, float* c)"
            print "{"
            print "  LIBXSMM_IMM(float, int, " + str(M) + ", " + str(N) + ", " + str(K) + ", a, b, c);"
            print "}"
            print
            print
            create_implementation("double", M, N, K, RowMajor, AlignedStores, AlignedLoads)
        else:
            mnklist = libxsmm_utilities.load_mnklist(sys.argv[5:])
            create_macros(RowMajor, AlignedStores, AlignedLoads, Alignment, mnklist, libxsmm_utilities.max_mnk(mnklist, Threshold))
    else:
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
