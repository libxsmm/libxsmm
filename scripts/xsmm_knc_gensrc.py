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


def create_macros(RowMajor, maxM, maxN, maxK):
    print "#define LIBXSMM_MAX_M " + str(maxM)
    print "#define LIBXSMM_MAX_N " + str(maxN)
    print "#define LIBXSMM_MAX_K " + str(maxK)
    if (0 != RowMajor):
        print "#define LIBXSMM_ROW_MAJOR 1"
        print "#define LIBXSMM_COL_MAJOR 0"
        print
        print "#define LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "  REAL alpha = 1, beta = 1; \\"
        print "  UINT m = M, n = N, k = K; \\"
        print "  char trans = 'N'; \\"
        print "  LIBXSMM_BLAS(REAL, gemm)(&trans, &trans, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n); \\"
        print "}"
    else:
        print "#define LIBXSMM_ROW_MAJOR 0"
        print "#define LIBXSMM_COL_MAJOR 1"
        print
        print "#define LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "  REAL alpha = 1, beta = 1; \\"
        print "  UINT m = M, n = N, k = K; \\"
        print "  char trans = 'N'; \\"
        print "  LIBXSMM_BLAS(REAL, gemm)(&trans, &trans, &m, &n, &k, &alpha, A, &k, B, &n, &beta, C, &n); \\"
        print "}"
    print
    print "#if !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)"
    print "# define LIBXSMM_SMM(REAL, UINT, M, N, K, A, B, C) LIBXSMM_BLASMM(REAL, UINT, M, N, K, A, B, C)"
    print "#else"
    if (0 != RowMajor):
        print "# define LIBXSMM_SMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "    LIBXSMM_PRAGMA(vector nontemporal(C)) \\"
        print "    LIBXSMM_PRAGMA(/*omp*/ simd collapse(2)) \\"
        print "    for (UINT j = 0; j < N; ++j) { \\"
        print "      for (UINT i = 0; i < M; ++i) { \\"
        print "        const UINT index = i * N + j; \\"
        print "        REAL r = C[index]; \\"
        print "        LIBXSMM_PRAGMA(unroll(16)) \\"
        print "        LIBXSMM_PRAGMA(/*omp*/ simd reduction(+:r)) \\"
        print "        for (UINT k = 0; k < K; ++k) { \\"
        print "          r += A[i*K+k] * B[k*N+j]; \\"
        print "        } \\"
        print "        C[index] = r; \\"
        print "      } \\"
        print "    } \\"
        print "  }"
    else:
        print "# define LIBXSMM_SMM(REAL, UINT, M, N, K, A, B, C) { \\"
        print "    LIBXSMM_PRAGMA(vector nontemporal(C)) \\"
        print "    LIBXSMM_PRAGMA(/*omp*/ simd collapse(2)) \\"
        print "    for (UINT j = 0; j < M; ++j) { \\"
        print "      for (UINT i = 0; i < N; ++i) { \\"
        print "        const UINT index = i * M + j; \\"
        print "        REAL r = C[index]; \\"
        print "        LIBXSMM_PRAGMA(unroll(16)) \\"
        print "        LIBXSMM_PRAGMA(/*omp*/ simd reduction(+:r)) \\"
        print "        for (UINT k = 0; k < K; ++k) { \\"
        print "          r += A[k*M+j] * B[i*K+k]; \\"
        print "        } \\"
        print "        C[index] = r; \\"
        print "      } \\"
        print "    } \\"
        print "  }"
    print "#endif"


def create_xsmm(RowMajor, M, N, K):
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
    print "#include \"xsmm_knc.h\""
    print "#include <xsmm_knc_util.h>"
    print
    print "#ifdef __cplusplus"
    print "extern \"C\" {"
    print "#endif"
    print
    print
    print "void dc_smm_dnn_" + str(M) + "_" + str(N) + "_" + str(K) + "(const double* a, const double* b, double* c)"
    print "{"
    print "#if defined(__MIC__)"
    print "  int i;"
    for mn in range(0, 8 * mnparts, 8):
        print "  {"
        mnm = min(mn + 7, Rows - 1)
        maskval = (1 << (mnm - mn + 1)) - 1
        print "    const __m512d x" + l1 + "[] = {"
        for k in range(0, K):
            print "      _MM512_MASK_LOADU_PD(" + l1 + " + " + str(Rows * k) + " + " + str(mn) + ", " + str(maskval) + "),"
        print "    };"
        print
        print "    for (i = 0; i < " + str(Cols) + "; ++i) {"
        print "      __m512d x" + l2 + "[" + str(K) + "], xc = _MM512_MASK_LOADU_PD(c + i * " + str(Rows) + " + " + str(mn) + ", " + str(maskval) + ");"
        for k in range(0, K):
            print "      x" + l2 + "[" + str(k) + "] = _mm512_set1_pd(" + l2 + "[i*" + str(K) + "+" + str(k) + "]);"
            print "      xc = _mm512_mask3_fmadd_pd(xa[" + str(k) + "], xb[" + str(k) + "], xc, " + str(maskval) + ");"
        print "      _MM512_MASK_STOREU_PD(c + i * " + str(Rows) + " + " + str(mn) + ", xc, " + str(maskval) + ");"
        print "    }"
        print "  }"
    print "#else"
    print "  LIBXSMM_SMM(double, int, "+ str(M) + ", " + str(N) + ", " + str(K) + ", a, b, c);"
    print "#endif"
    print "}"
    print
    print
    print "#ifdef __cplusplus"
    print "} // extern \"C\""
    print "#endif"


def load_dims(dims):
    dims = map(int, dims) #; dims.sort()
    return list(set(dims))


if (6 <= len(sys.argv)):
    RowMajor = int(sys.argv[1])

    if (0 != int(sys.argv[2])):
        create_xsmm(RowMajor, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif (7 <= len(sys.argv)):
        dimsM = load_dims(sys.argv[5:5+int(sys.argv[3])])
        dimsN = load_dims(sys.argv[5+int(sys.argv[3]):5+int(sys.argv[3])+int(sys.argv[4])])
        dimsK = load_dims(sys.argv[5+int(sys.argv[3])+int(sys.argv[4]):])
        create_macros(RowMajor, max(dimsM), max(dimsN), max(dimsK))
    else:
        sys.stderr.write(sys.argv[0] + ": wrong number of arguments!\n")
else:
    sys.stderr.write(sys.argv[0] + ": wrong number of arguments!\n")
