###############################################################################
## Copyright (c) 2013-2014, Intel Corporation                                ##
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


def create_macros(RowMajor, maxMNK):
    print "#define LIBXSMM_MAX_MNK " + str(maxMNK)
    if (0 != RowMajor):
        print "#define LIBXSMM_ROW_MAJOR 1"
        print "#define LIBXSMM_COL_MAJOR 0"
        print
        print "#define LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "  REAL libxsmm_alpha_ = 1, libxsmm_beta_ = 1; \\"
        print "  UINT libxsmm_m_ = (M), libxsmm_n_ = (N), libxsmm_k_ = (K); \\"
        print "  char libxsmm_trans_ = 'N'; \\"
        print "  LIBXSMM_BLASPREC(, REAL, gemm)(&libxsmm_trans_, &libxsmm_trans_, \\"
        print "    &libxsmm_n_, &libxsmm_m_, &libxsmm_k_, \\"
        print "    &libxsmm_alpha_, (REAL*)(B), &libxsmm_n_, (REAL*)(A), &libxsmm_k_, \\"
        print "    &libxsmm_beta_, (C), &libxsmm_n_); \\"
        print "}"
    else:
        print "#define LIBXSMM_ROW_MAJOR 0"
        print "#define LIBXSMM_COL_MAJOR 1"
        print
        print "#define LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "  UINT libxsmm_m_ = (M), libxsmm_n_ = (N), libxsmm_k_ = (K); \\"
        print "  REAL libxsmm_alpha_ = 1, libxsmm_beta_ = 1; \\"
        print "  char libxsmm_trans_ = 'N'; \\"
        print "  LIBXSMM_BLASPREC(, REAL, gemm)(&libxsmm_trans_, &libxsmm_trans_, \\"
        print "    &libxsmm_m_, &libxsmm_n_, &libxsmm_k_, \\"
        print "    &libxsmm_alpha_, (REAL*)(A), &libxsmm_k_, (REAL*)(B), &libxsmm_n_, \\"
        print "    &libxsmm_beta_, (C), &libxsmm_n_); \\"
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


def create_mm(Real, RowMajor, M, N, K):
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
        for k in range(0, K):
            print "      MM512_MASK_LOADU_PD(" + l1 + " + " + str(Rows * k) + " + " + str(mn) + ", " + str(maskval) + "),"
        print "    };"
        print
        print "    for (i = 0; i < " + str(Cols) + "; ++i) {"
        print "      const int index = i * " + str(Rows) + " + " + str(mn) + ";"
        print "      __m512" + make_typepfix(Real) + " x" + l2 + "[" + str(K) + "], xc = MM512_MASK_LOADU_PD(c + index, " + str(maskval) + ");"
        for k in range(0, K):
            print "      x" + l2 + "[" + str(k) + "] = _mm512_set1_pd(" + l2 + "[i*" + str(K) + "+" + str(k) + "]);"
            print "      xc = _mm512_mask3_fmadd_pd(xa[" + str(k) + "], xb[" + str(k) + "], xc, " + str(maskval) + ");"
        print "      MM512_MASK_STOREU_PD(c + index, xc, " + str(maskval) + ");"
        print "    }"
        print "  }"
    print "#else"
    print "  LIBXSMM_SMM(" + Real + ", int, " + str(M) + ", " + str(N) + ", " + str(K) + ", a, b, c);"
    print "#endif"
    print "}"


def load_dims(dims):
    dims = list(map(int, dims)) #; dims.sort()
    return dims


if (6 <= len(sys.argv)):
    RowMajor = int(sys.argv[1])

    if (0 > int(sys.argv[2])):
        print "#include \"libxsmm.h\""
        print "#include <libxsmm_knc.h>"
        print
        print
        M = int(sys.argv[3])
        N = int(sys.argv[4])
        K = int(sys.argv[5])
        # Note: create_mm is not yet ready to generate the single-precision implementation
        print "LIBXSMM_EXTERN_C void libxsmm_smm_" + str(M) + "_" + str(N) + "_" + str(K) + "(const float* a, const float* b, float* c)"
        print "{"
        print "  LIBXSMM_SMM(float, int, " + str(M) + ", " + str(N) + ", " + str(K) + ", a, b, c);"
        print "}"
        print
        print
        create_mm("double", RowMajor, M, N, K)
    elif (7 <= len(sys.argv)):
        dimsM = load_dims(sys.argv[5:5+int(sys.argv[3])])
        dimsN = load_dims(sys.argv[5+int(sys.argv[3]):5+int(sys.argv[3])+int(sys.argv[4])])
        dimsK = load_dims(sys.argv[5+int(sys.argv[3])+int(sys.argv[4]):])
        maxMNK = int(sys.argv[2])
        for m in dimsM:
            for n in dimsN:
                for k in dimsK:
                    maxMNK = max(maxMNK, m * n * k)
        create_macros(RowMajor, maxMNK)
    else:
        sys.stderr.write(sys.argv[0] + ": wrong number of arguments!\n")
else:
    sys.stderr.write(sys.argv[0] + ": wrong number of arguments!\n")
