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
## Hans Pabst (Intel Corp.)
###############################################################################
import libxsmm_utilities
import math, sys


if __name__ == "__main__":
    argc = len(sys.argv)
    if (6 == argc):
        row_major = int(sys.argv[1])
        m, n, k = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        prefetch = int(sys.argv[5])

        mnkstr = str(m) + "_" + str(n) + "_" + str(k)
        if (0 != row_major):
            signature = "b, a, c, pb, pa, pc" if (0 != prefetch) else "b, a, c"
        else: # ColMajor
            signature = "a, b, c, pa, pb, pc" if (0 != prefetch) else "a, b, c"
        print
        print "LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_smm_" + mnkstr + "("
        print "  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c" \
           + (",\n  const float* pa, const float* pb, const float* pc)" if (0 != prefetch) else ")")
        print "{"
        print "#if defined(__AVX512F__) && defined(LIBXSMM_GENTARGET_knl_sp)"
        print "  libxsmm_smm_" + mnkstr + "_knl(" + signature + ");"
        print "#elif defined(__AVX2__) && defined(LIBXSMM_GENTARGET_hsw_sp)"
        print "  libxsmm_smm_" + mnkstr + "_hsw(" + signature + ");"
        print "#elif defined(__AVX__) && defined(LIBXSMM_GENTARGET_snb_sp)"
        print "  libxsmm_smm_" + mnkstr + "_snb(" + signature + ");"
        print "#elif defined(__SSE3__) && defined(LIBXSMM_GENTARGET_wsm_sp)"
        print "  libxsmm_smm_" + mnkstr + "_wsm(" + signature + ");"
        print "#elif defined(__MIC__) && defined(LIBXSMM_GENTARGET_knc_sp)"
        print "  libxsmm_smm_" + mnkstr + "_knc(" + signature + ");"
        print "#else"
        print "  LIBXSMM_IMM(float, int, LIBXSMM_FLAGS, " + str(m) + ", " + str(n) + ", " + str(k) + ", a, b, c" + (", pa, pb, pc);" if (0 != prefetch) else ");")
        print "#endif"
        print "}"
        print
        print
        print "LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_dmm_" + mnkstr + "("
        print "  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c" \
           + (",\n  const double* pa, const double* pb, const double* pc)" if (0 != prefetch) else ")")
        print "{"
        print "#if defined(__AVX512F__) && defined(LIBXSMM_GENTARGET_knl_dp)"
        print "  libxsmm_dmm_" + mnkstr + "_knl(" + signature + ");"
        print "#elif defined(__AVX2__) && defined(LIBXSMM_GENTARGET_hsw_dp)"
        print "  libxsmm_dmm_" + mnkstr + "_hsw(" + signature + ");"
        print "#elif defined(__AVX__) && defined(LIBXSMM_GENTARGET_snb_dp)"
        print "  libxsmm_dmm_" + mnkstr + "_snb(" + signature + ");"
        print "#elif defined(__SSE3__) && defined(LIBXSMM_GENTARGET_wsm_dp)"
        print "  libxsmm_dmm_" + mnkstr + "_wsm(" + signature + ");"
        print "#elif defined(__MIC__) && defined(LIBXSMM_GENTARGET_knc_dp)"
        print "  libxsmm_dmm_" + mnkstr + "_knc(" + signature + ");"
        print "#else"
        print "  LIBXSMM_IMM(double, int, LIBXSMM_FLAGS, " + str(m) + ", " + str(n) + ", " + str(k) + ", a, b, c" + (", pa, pb, pc);" if (0 != prefetch) else ");")
        print "#endif"
        print "}"
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
