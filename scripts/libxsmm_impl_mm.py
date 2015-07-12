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
    print "#define LIBXSMM_ALIGNED_STORES " + str(AlignedStores)
    print "#define LIBXSMM_ALIGNED_LOADS " + str(AlignedLoads)
    print "#define LIBXSMM_ALIGNED_MAX " + str(max(Alignment, AlignedStores, AlignedLoads))
    print "#define LIBXSMM_ROW_MAJOR " + ["0", "1"][0 != RowMajor]
    print "#define LIBXSMM_COL_MAJOR " + ["1", "0"][0 != RowMajor]
    maxMNK = libxsmm_utilities.max_mnk(listMNK, Threshold)
    avgDim = int(maxMNK ** (1.0 / 3.0) + 0.5)
    maxM = libxsmm_utilities.max_mnk(listMNK, avgDim, 0)
    maxN = libxsmm_utilities.max_mnk(listMNK, avgDim, 1)
    maxK = libxsmm_utilities.max_mnk(listMNK, avgDim, 2)
    print "#define LIBXSMM_MAX_MNK " + str(maxMNK)
    print "#define LIBXSMM_MAX_M " + str(maxM)
    print "#define LIBXSMM_MAX_N " + str(maxN)
    print "#define LIBXSMM_MAX_K " + str(maxK)
    listM = libxsmm_utilities.make_mlist(listMNK); listM.append(avgDim)
    listN = libxsmm_utilities.make_nlist(listMNK); listN.append(avgDim)
    listK = libxsmm_utilities.make_klist(listMNK); listK.append(avgDim)
    print "#define LIBXSMM_AVG_M " + str(libxsmm_utilities.median(listM))
    print "#define LIBXSMM_AVG_N " + str(libxsmm_utilities.median(listN))
    print "#define LIBXSMM_AVG_K " + str(libxsmm_utilities.median(listK))


def declare_variables(name, n, unroll):
    result = ""
    for i in range(0, min(unroll, n)): result += [name, ", " + name][0<i] + str(i)
    if (0 < (n - unroll)): result = [name, ", " + name][0<unroll] + "[" + str(n - unroll) + "]"
    return result


def get_var(name, i, unroll):
    if (0 <= (i - unroll)):
        return name + "[" + str(i) + "]"
    else:
        return name + str(i)


def create_implementation(Real, M, N, K, RowMajor, AlignedStores, AlignedLoads, Unroll):
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
    mnk = str(M) + "_" + str(N) + "_" + str(K)
    print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_" + libxsmm_utilities.make_typeflag(Real) + "mm_" + mnk + "(const " + Real + "* a, const " + Real + "* b, " + Real + "* c)"
    print "{"
    print "#if defined(__MIC__) || defined(__AVX512F__)"
    if (1 < AlignedStores):
        print "  const int r = LIBXSMM_ALIGN_VALUE(int, " + Real + ", " + str(Rows) + ", LIBXSMM_ALIGNED_STORES);"
    else:
        print "  const int r = " + str(Rows) + ";"
    if (0 != Unroll):
        print "  int i = 0;"
    else:
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
        print "    __m512" + libxsmm_utilities.make_typepfix(Real) + " xc = MM512_LOAD" + ["U", ""][1 < AlignedLoads] + mask_inst + "_PD(dst" + mask_argv + ", _MM_HINT_NONE);"
        print "    __m512" + libxsmm_utilities.make_typepfix(Real) + " " + declare_variables("x" + l1, K, Unroll - 2) + ";"
        print "    __m512" + libxsmm_utilities.make_typepfix(Real) + " " + declare_variables("x" + l2, K, Unroll - 2) + ";"
        print
        if (1 <= Unroll):
            for k in range(0, K):
                print "    " + get_var("x" + l1, k, Unroll - 2) + " = MM512_LOAD" + ["U", ""][1 < AlignedLoads] + mask_inst + "_PD(" + l1 + " + " + str(k * Rows) + " + " + str(mn) + mask_argv + ", _MM_HINT_NONE),"
                print "    " + get_var("x" + l2, k, Unroll - 2) + " = MM512_SET1_PD(src[" + str(k) + "]);"
                print "    xc = MM512_FMADD" + mask_inst +"_PD(" + get_var("xa", k, Unroll - 2) + ", " + get_var("xb", k, Unroll - 2) + ", xc" + mask_argv + ");"
        else:
            print "    for (k = 0; k < " + str(K) + "; ++k) {"
            print "      x" + l1 + "[k] = MM512_LOAD" + ["U", ""][1 < AlignedLoads] + mask_inst + "_PD(" + l1 + " + k * " + str(Rows) + " + " + str(mn) + mask_argv + ", _MM_HINT_NONE),"
            print "      x" + l2 + "[k] = MM512_SET1_PD(src[k]);"
            print "      xc = MM512_FMADD" + mask_inst +"_PD(xa[k], xb[k], xc" + mask_argv + ");"
            print "    }"
        print "    MM512_STORE" + ["U", ""][1 < AlignedStores] + mask_inst + "_PD(dst, xc" + mask_argv + ", _MM_HINT_NONE);"
        print
        print "    for (i = 1; i < " + str(Cols) + "; ++i) {"
        print "      src += " + str(K) + "; dst += r;"
        print "      xc = MM512_LOAD" + ["U", ""][1 < AlignedLoads] + mask_inst + "_PD(dst" + mask_argv + ", _MM_HINT_NONE);"
        if (2 <= Unroll):
            for k in range(0, K):
                print "      " + get_var("x" + l2, k, Unroll - 2) + " = MM512_SET1_PD(src[" + str(k) + "]);"
                print "      xc = MM512_FMADD" + mask_inst +"_PD(" + get_var("xa", k, Unroll -2) + ", " + get_var("xb", k, Unroll -2) + ", xc" + mask_argv + ");"
        else:
            print
            print "      for (k = 0; k < " + str(K) + "; ++k) {"
            print "        x" + l2 + "[k] = MM512_SET1_PD(src[k]);"
            print "        xc = MM512_FMADD" + mask_inst +"_PD(xa[k], xb[k], xc" + mask_argv + ");"
            print "      }"
        print "      MM512_STORE" + ["U", ""][1 < AlignedStores] + mask_inst + "_PD(dst, xc" + mask_argv + ", _MM_HINT_NONE);"
        print "    }"
        print "  }"
    print "#else"
    print "  LIBXSMM_IMM(" + Real + ", int, " + str(M) + ", " + str(N) + ", " + str(K) + ", a, b, c);"
    print "#endif"
    print "}"


def create_gentarget(m, n, k, row_major):
    mnk = str(m) + "_" + str(n) + "_" + str(k)
    if (0 != row_major):
        a, b = "b", "a"
    else: # ColMajor
        a, b = "a", "b"
    print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_dmm_" + mnk + "(const double* a, const double* b, double* c)"
    print "{"
    print "#if defined(__AVX512F__) && defined(LIBXSMM_GENTARGET_knl_dp)"
    print "  libxsmm_dmm_" + mnk + "_knl(" + a + ", " + b + ", c);"
    print "#elif defined(__AVX2__) && defined(LIBXSMM_GENTARGET_hsw_dp)"
    print "  libxsmm_dmm_" + mnk + "_hsw(" + a + ", " + b + ", c);"
    print "#elif defined(__AVX__) && defined(LIBXSMM_GENTARGET_snb_dp)"
    print "  libxsmm_dmm_" + mnk + "_snb(" + a + ", " + b + ", c);"
    print "#elif defined(__SSE3__) && defined(LIBXSMM_GENTARGET_wsm_dp)"
    print "  libxsmm_dmm_" + mnk + "_wsm(" + a + ", " + b + ", c);"
    print "#elif defined(__MIC__) && defined(LIBXSMM_GENTARGET_knc_dp)"
    print "  libxsmm_dmm_" + mnk + "_knc(" + a + ", " + b + ", c);"
    print "#else"
    print "  LIBXSMM_IMM(double, int, " + str(m) + ", " + str(n) + ", " + str(k) + ", a, b, c);"
    print "#endif"
    print "}"
    print
    print
    print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_smm_" + mnk + "(const float* a, const float* b, float* c)"
    print "{"
    print "#if defined(__AVX512F__) && defined(LIBXSMM_GENTARGET_knl_sp)"
    print "  libxsmm_smm_" + mnk + "_knl(" + a + ", " + b + ", c);"
    print "#elif defined(__AVX2__) && defined(LIBXSMM_GENTARGET_hsw_sp)"
    print "  libxsmm_smm_" + mnk + "_hsw(" + a + ", " + b + ", c);"
    print "#elif defined(__AVX__) && defined(LIBXSMM_GENTARGET_snb_sp)"
    print "  libxsmm_smm_" + mnk + "_snb(" + a + ", " + b + ", c);"
    print "#elif defined(__SSE3__) && defined(LIBXSMM_GENTARGET_wsm_sp)"
    print "  libxsmm_smm_" + mnk + "_wsm(" + a + ", " + b + ", c);"
    print "#elif defined(__MIC__) && defined(LIBXSMM_GENTARGET_knc_sp)"
    print "  libxsmm_smm_" + mnk + "_knc(" + a + ", " + b + ", c);"
    print "#else"
    print "  LIBXSMM_IMM(float, int, " + str(m) + ", " + str(n) + ", " + str(k) + ", a, b, c);"
    print "#endif"
    print "}"


if __name__ == '__main__':
    argc = len(sys.argv)
    if (5 < argc):
        RowMajor = int(sys.argv[1])
        AlignedStores = libxsmm_utilities.sanitize_alignment(int(sys.argv[2]))
        AlignedLoads = libxsmm_utilities.sanitize_alignment(int(sys.argv[3]))
        Alignment = libxsmm_utilities.sanitize_alignment(int(sys.argv[4]))
        Threshold = int(sys.argv[5])

        if (-1 > Threshold): # implementation (translation unit)
            print "#include <libxsmm_isa.h>"
            print "#include <libxsmm.h>"
            print
            print
            m, n, k = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
            mnk = str(m) + "_" + str(n) + "_" + str(k)
            # Note: create_implementation is not yet ready to generate the single-precision implementation
            print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_smm_" + mnk + "(const float* a, const float* b, float* c)"
            print "{"
            print "  LIBXSMM_IMM(float, int, " + str(m) + ", " + str(n) + ", " + str(k) + ", a, b, c);"
            print "}"
            print
            print
            create_implementation("double", m, n, k, RowMajor, AlignedStores, AlignedLoads, -1 * (Threshold + 1))
        elif (-1 == Threshold): # static ISA dispatch (translation unit)
            print
            m, n, k = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
            create_gentarget(m, n, k, RowMajor)
        else: # macros (header based implementation)
            mnklist = libxsmm_utilities.load_mnklist(sys.argv[6:], 0, Threshold)
            create_macros(RowMajor, AlignedStores, AlignedLoads, Alignment, mnklist, Threshold)
    else:
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
