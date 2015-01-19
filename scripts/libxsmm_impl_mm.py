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
import math
import sys


def create_macros(RowMajor, AlignedStores, AlignedLoads, maxMNK):
    print "#define LIBXSMM_MAX_MNK " + str(maxMNK)
    print "#define LIBXSMM_ALIGNED_STORES " + ["0", "1"][0 != AlignedStores]
    print "#define LIBXSMM_ALIGNED_LOADS " + ["0", "1"][0 != AlignedLoads]
    print "#define LIBXSMM_ROW_MAJOR " + ["0", "1"][0 != RowMajor]
    print "#define LIBXSMM_COL_MAJOR " + ["1", "0"][0 != RowMajor]
    print
    if (0 != RowMajor):
        print "#define LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "  REAL libxsmm_alpha_ = 1, libxsmm_beta_ = 1; \\"
        print "  UINT libxsmm_m_ = (M), libxsmm_n_ = (N), libxsmm_k_ = (K); \\"
        print "  char libxsmm_trans_ = 'N'; \\"
        print "  LIBXSMM_FSYMBOL(LIBXSMM_BLASPREC(, REAL, gemm))(&libxsmm_trans_, &libxsmm_trans_, \\"
        print "    &libxsmm_n_, &libxsmm_m_, &libxsmm_k_, \\"
        print "    &libxsmm_alpha_, (REAL*)(B), &libxsmm_n_, (REAL*)(A), &libxsmm_k_, \\"
        print "    &libxsmm_beta_, (C), &libxsmm_n_); \\"
        print "}"
    else:
        print "#define LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "  UINT libxsmm_m_ = (M), libxsmm_n_ = (N), libxsmm_k_ = (K); \\"
        print "  REAL libxsmm_alpha_ = 1, libxsmm_beta_ = 1; \\"
        print "  char libxsmm_trans_ = 'N'; \\"
        print "  LIBXSMM_FSYMBOL(LIBXSMM_BLASPREC(, REAL, gemm))(&libxsmm_trans_, &libxsmm_trans_, \\"
        print "    &libxsmm_m_, &libxsmm_n_, &libxsmm_k_, \\"
        print "    &libxsmm_alpha_, (REAL*)(A), &libxsmm_m_, (REAL*)(B), &libxsmm_k_, \\"
        print "    &libxsmm_beta_, (C), &libxsmm_m_); \\"
        print "}"
    print
    print "#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)"
    print "# define LIBXSMM_SMM(REAL, UINT, M, N, K, A, B, C) LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C)"
    print "#else"
    if (0 != RowMajor):
        print "# define LIBXSMM_SMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "    UINT libxsmm_i_, libxsmm_j_, libxsmm_k_; \\"
        print "    REAL *const libxsmm_c_ = (C); \\"
        print "    LIBXSMM_PRAGMA(vector nontemporal(libxsmm_c_)) \\"
        print "    LIBXSMM_PRAGMA(/*omp*/ simd collapse(2)) \\"
        print "    for (libxsmm_j_ = 0; libxsmm_j_ < (N); ++libxsmm_j_) { \\"
        print "      for (libxsmm_i_ = 0; libxsmm_i_ < (M); ++libxsmm_i_) { \\"
        print "        const UINT libxsmm_index_ = libxsmm_i_ * (N) + libxsmm_j_; \\"
        print "        REAL libxsmm_r_ = libxsmm_c_[libxsmm_index_]; \\"
        print "        LIBXSMM_PRAGMA(unroll(16)) \\"
        print "        LIBXSMM_PRAGMA(/*omp*/ simd reduction(+:libxsmm_r_)) \\"
        print "        for (libxsmm_k_ = 0; libxsmm_k_ < (K); ++libxsmm_k_) { \\"
        print "          libxsmm_r_ += (A)[libxsmm_i_*K+libxsmm_k_] * (B)[libxsmm_k_*N+libxsmm_j_]; \\"
        print "        } \\"
        print "        libxsmm_c_[libxsmm_index_] = libxsmm_r_; \\"
        print "      } \\"
        print "    } \\"
        print "  }"
    else:
        print "# define LIBXSMM_SMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "    UINT libxsmm_i_, libxsmm_j_, libxsmm_k_; \\"
        print "    REAL *const libxsmm_c_ = (C); \\"
        print "    LIBXSMM_PRAGMA(vector nontemporal(libxsmm_c_)) \\"
        print "    LIBXSMM_PRAGMA(/*omp*/ simd collapse(2)) \\"
        print "    for (libxsmm_j_ = 0; libxsmm_j_ < (M); ++libxsmm_j_) { \\"
        print "      for (libxsmm_i_ = 0; libxsmm_i_ < (N); ++libxsmm_i_) { \\"
        print "        const UINT libxsmm_index_ = libxsmm_i_ * (M) + libxsmm_j_; \\"
        print "        REAL libxsmm_r_ = libxsmm_c_[libxsmm_index_]; \\"
        print "        LIBXSMM_PRAGMA(unroll(16)) \\"
        print "        LIBXSMM_PRAGMA(/*omp*/ simd reduction(+:libxsmm_r_)) \\"
        print "        for (libxsmm_k_ = 0; libxsmm_k_ < (K); ++libxsmm_k_) { \\"
        print "          libxsmm_r_ += (A)[libxsmm_k_*M+libxsmm_j_] * (B)[libxsmm_i_*K+libxsmm_k_]; \\"
        print "        } \\"
        print "        libxsmm_c_[libxsmm_index_] = libxsmm_r_; \\"
        print "      } \\"
        print "    } \\"
        print "  }"
    print "#endif"


def make_typeflag(Real):
    return ["s", "d"]["float" != Real]


def make_typepfix(Real):
    return ["", "d"]["float" != Real]


def create_implementation(Real, M, N, K, RowMajor, AlignedStores, AlignedLoads):
    if (0 != RowMajor):
        Rows, Cols = N, M
        l1, l2 = "b", "a"
    else:
        Rows, Cols = M, N
        l1, l2 = "a", "b"
    iparts = int(math.floor(Rows / 8))
    if (0 == (Rows % 8)):
        mnparts = iparts
    else:
        mnparts = iparts + 1
    print "LIBXSMM_EXTERN_C void libxsmm_" + make_typeflag(Real) + "mm_" + str(M) + "_" + str(N) + "_" + str(K) + "(const " + Real + "* a, const " + Real + "* b, " + Real + "* c)"
    print "{"
    print "#if defined(__MIC__)"
    print "  int i;"
    for mn in range(0, 8 * mnparts, 8):
        print "  {"
        mnm = min(mn + 7, Rows - 1)
        maskval = (1 << (mnm - mn + 1)) - 1
        print "    const __m512" + make_typepfix(Real) + " x" + l1 + "[] = {"
        if (255 != maskval):
            mask_inst, mask_argv, mask3 = "_MASK", ", " + str(maskval), "_mask3"
        else:
            mask_inst, mask_argv, mask3 = "", "", ""
        for k in range(0, K):
            print "      MM512_LOAD" + ["U", ""][0 != AlignedLoads] + mask_inst + "_PD(" + l1 + " + " + str(Rows * k) + " + " + str(mn) + mask_argv + "),"
        print "    };"
        print
        print "    for (i = 0; i < " + str(Cols) + "; ++i) {"
        print "      const int index = i * " + str(Rows) + " + " + str(mn) + ";"
        print "      __m512" + make_typepfix(Real) + " xc = MM512_LOADNT" + ["U", ""][0 != AlignedLoads] + mask_inst + "_PD(c + index" + mask_argv + "), x" + l2 + "[" + str(K) + "];"
        for k in range(0, K):
            print "      x" + l2 + "[" + str(k) + "] = _mm512_set1_pd(" + l2 + "[i*" + str(K) + "+" + str(k) + "]);"
            print "      xc = _mm512" + mask3 + "_fmadd_pd(xa[" + str(k) + "], xb[" + str(k) + "], xc" + mask_argv + ");"
        print "      MM512_STORENT" + ["U", ""][0 != AlignedStores] + mask_inst + "_PD(c + index, xc" + mask_argv + ");"
        print "    }"
        print "  }"
    print "#else"
    print "  LIBXSMM_SMM(" + Real + ", int, " + str(M) + ", " + str(N) + ", " + str(K) + ", a, b, c);"
    print "#endif"
    print "}"


def load_dims(dims):
    dims = list(map(int, dims)) #; dims.sort()
    return dims


if (7 <= len(sys.argv)):
    RowMajor = int(sys.argv[1])
    AlignedStores = int(sys.argv[2])
    AlignedLoads = int(sys.argv[3])
    Threshold = int(sys.argv[4])

    if (0 > Threshold):
        print "#include <libxsmm_isa.h>"
        print
        print
        M = int(sys.argv[5])
        N = int(sys.argv[6])
        K = int(sys.argv[7])
        # Note: create_implementation is not yet ready to generate the single-precision implementation
        print "LIBXSMM_EXTERN_C void libxsmm_smm_" + str(M) + "_" + str(N) + "_" + str(K) + "(const float* a, const float* b, float* c)"
        print "{"
        print "  LIBXSMM_SMM(float, int, " + str(M) + ", " + str(N) + ", " + str(K) + ", a, b, c);"
        print "}"
        print
        print
        create_implementation("double", M, N, K, RowMajor, AlignedStores, AlignedLoads)
    elif (9 <= len(sys.argv)):
        maxMNK = Threshold
        dimsM = load_dims(sys.argv[7:7+int(sys.argv[5])])
        dimsN = load_dims(sys.argv[7+int(sys.argv[5]):7+int(sys.argv[5])+int(sys.argv[6])])
        dimsK = load_dims(sys.argv[7+int(sys.argv[5])+int(sys.argv[6]):])
        for m in dimsM:
            for n in dimsN:
                for k in dimsK:
                    maxMNK = max(maxMNK, m * n * k)
        create_macros(RowMajor, AlignedStores, AlignedLoads, maxMNK)
    else:
        sys.stderr.write(sys.argv[0] + ": wrong number of arguments!\n")
else:
    sys.stderr.write(sys.argv[0] + ": wrong number of arguments!\n")
