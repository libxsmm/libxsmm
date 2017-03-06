#!/usr/bin/env python
###############################################################################
# Copyright (c) 2014-2017, Intel Corporation                                  #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions          #
# are met:                                                                    #
# 1. Redistributions of source code must retain the above copyright           #
#    notice, this list of conditions and the following disclaimer.            #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR       #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT        #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################
import sys


if __name__ == "__main__":
    argc = len(sys.argv)
    if (6 == argc):
        precision = int(sys.argv[1])
        m, n, k = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        prefetch = int(sys.argv[5])

        mnkstr = str(m) + "_" + str(n) + "_" + str(k)
        signature = ["a, b, c, pa, pb, pc", "a, b, c"][0 >= prefetch]
        if (2 != precision):
            pfsig = [")",
                     ", const float* pa"
                     ", const float* pb"
                     ", const float* pc)"][0 < prefetch]
            print
            print
            print("LIBXSMM_API_DEFINITION void libxsmm_smm_" + mnkstr + "(")
            print("  const float* a, const float* b, float* c" + pfsig)
            print("{")
            print("#if defined(__AVX512F__) && "
                  "defined(LIBXSMM_GENTARGET_knl_sp)")
            print("  libxsmm_smm_" + mnkstr + "_knl(" + signature + ");")
            print("#elif defined(__AVX2__) && "
                  "defined(LIBXSMM_GENTARGET_hsw_sp)")
            print("  libxsmm_smm_" + mnkstr + "_hsw(" + signature + ");")
            print("#elif defined(__AVX__) && "
                  "defined(LIBXSMM_GENTARGET_snb_sp)")
            print("  libxsmm_smm_" + mnkstr + "_snb(" + signature + ");")
            print("#elif defined(__SSE3__) && "
                  "defined(LIBXSMM_GENTARGET_wsm_sp)")
            print("  libxsmm_smm_" + mnkstr + "_wsm(" + signature + ");")
            print("#elif defined(__MIC__) && "
                  "defined(LIBXSMM_GENTARGET_knc_sp)")
            print("  libxsmm_smm_" + mnkstr + "_knc(" + signature + ");")
            print("#else")
            print("  LIBXSMM_MESSAGE(\"================================"
                  "================================================\")")
            print("  LIBXSMM_MESSAGE(\"LIBXSMM: No specific instruction"
                  " set extension found for specialization!\")")
            print("  LIBXSMM_MESSAGE(\"================================"
                  "================================================\")")
            if (0 < prefetch):
                print("  LIBXSMM_UNUSED(pa);"
                      " LIBXSMM_UNUSED(pb);"
                      " LIBXSMM_UNUSED(pc);")
            print("  LIBXSMM_INLINE_XGEMM(float, "
                  "int/*libxsmm_blasint not req.*/,\n"
                  "    LIBXSMM_FLAGS, "
                  "LIBXSMM_LD(" + str(m) + ", " + str(n) + "), "
                  "LIBXSMM_LD(" + str(n) + ", " + str(m) + "), " +
                  str(k) + ",\n"
                  "    LIBXSMM_ALPHA, LIBXSMM_LD(a, b), "
                  "LIBXSMM_LD(" + str(m) + ", " + str(n) + "), "
                  "LIBXSMM_LD(b, a), " + str(k) + ",\n"
                  "    LIBXSMM_BETA, c, "
                  "LIBXSMM_LD(" + str(m) + ", " + str(n) + "));")
            print("#endif")
            print("}")
        if (1 != precision):
            pfsig = [")",
                     ", const double* pa"
                     ", const double* pb"
                     ", const double* pc)"][0 < prefetch]
            print
            print
            print("LIBXSMM_API_DEFINITION void libxsmm_dmm_" + mnkstr + "(")
            print("  const double* a, const double* b, double* c" + pfsig)
            print("{")
            print("#if defined(__AVX512F__) && "
                  "defined(LIBXSMM_GENTARGET_knl_dp)")
            print("  libxsmm_dmm_" + mnkstr + "_knl(" + signature + ");")
            print("#elif defined(__AVX2__) && "
                  "defined(LIBXSMM_GENTARGET_hsw_dp)")
            print("  libxsmm_dmm_" + mnkstr + "_hsw(" + signature + ");")
            print("#elif defined(__AVX__) && "
                  "defined(LIBXSMM_GENTARGET_snb_dp)")
            print("  libxsmm_dmm_" + mnkstr + "_snb(" + signature + ");")
            print("#elif defined(__SSE3__) && "
                  "defined(LIBXSMM_GENTARGET_wsm_dp)")
            print("  libxsmm_dmm_" + mnkstr + "_wsm(" + signature + ");")
            print("#elif defined(__MIC__) && "
                  "defined(LIBXSMM_GENTARGET_knc_dp)")
            print("  libxsmm_dmm_" + mnkstr + "_knc(" + signature + ");")
            print("#else")
            print("  /** Generate below message once per translation unit "
                  "(already emitted for single-precision; see above)")
            print("  LIBXSMM_MESSAGE(\"No specific instruction set extension"
                  " found for specialization!\")*/")
            if (0 < prefetch):
                print("  LIBXSMM_UNUSED(pa);"
                      " LIBXSMM_UNUSED(pb);"
                      " LIBXSMM_UNUSED(pc);")
            print("  LIBXSMM_INLINE_XGEMM(double, "
                  "int/*libxsmm_blasint not req.*/,\n"
                  "    LIBXSMM_FLAGS, "
                  "LIBXSMM_LD(" + str(m) + ", " + str(n) + "), "
                  "LIBXSMM_LD(" + str(n) + ", " + str(m) + "), " +
                  str(k) + ",\n"
                  "    LIBXSMM_ALPHA, LIBXSMM_LD(a, b), "
                  "LIBXSMM_LD(" + str(m) + ", " + str(n) + "), "
                  "LIBXSMM_LD(b, a), " + str(k) + ",\n"
                  "    LIBXSMM_BETA, c, "
                  "LIBXSMM_LD(" + str(m) + ", " + str(n) + "));")
            print("#endif")
            print("}")
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
