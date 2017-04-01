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
import libxsmm_utilities
import sys


if __name__ == "__main__":
    argc = len(sys.argv)
    if (3 < argc):
        precision = int(sys.argv[1])
        threshold = int(sys.argv[2])
        mnklist = libxsmm_utilities.load_mnklist(sys.argv[3:], 0)

        print("libxsmm_gemm_descriptor desc;")
        print("libxsmm_xmmfunction func;")
        print("unsigned int hash, indx;")
        print("#if defined(_MSC_VER)")
        print("# pragma warning(push)")
        print("# pragma warning(disable: 4127)")
        print("#endif")
        for mnk in mnklist:
            mstr, nstr, kstr, mnkstr = \
                str(mnk[0]), str(mnk[1]), str(mnk[2]), "_".join(map(str, mnk))
            mnksig = ("LIBXSMM_LD(" + mstr + ", " + nstr + "), " +
                      "LIBXSMM_LD(" + nstr + ", " + mstr + "), " + kstr)
            ldxsig = ("LIBXSMM_LD(" + mstr + ", " + nstr + "), " +
                      kstr + ", " + "LIBXSMM_LD(" + mstr + ", " + nstr + ")")
            # prefer registering double-precision kernels
            # when approaching an exhausted registry
            if (1 != precision):  # only double-precision
                print("if (LIBXSMM_GEMM_NO_BYPASS_DIMS(" + mnksig + ") &&")
                print("    LIBXSMM_GEMM_NO_BYPASS_DIMS(" + ldxsig + ")) {")
                print("  LIBXSMM_GEMM_DESCRIPTOR(desc, " +
                      "LIBXSMM_ALIGNMENT, LIBXSMM_FLAGS,")
                print("    " + mnksig + ", " + ldxsig + ",")
                print("    LIBXSMM_ALPHA, LIBXSMM_BETA, INTERNAL_PREFETCH);")
                print("    LIBXSMM_HASH_FUNCTION_CALL(hash, indx, &desc);" +
                      " func.dmm = (libxsmm_dmmfunction)libxsmm_dmm_" +
                      mnkstr + ";")
                print("  internal_register_static_code(" +
                      "&desc, indx, hash, func, result);")
                print("}")
        for mnk in mnklist:
            mstr, nstr, kstr, mnkstr = \
                str(mnk[0]), str(mnk[1]), str(mnk[2]), "_".join(map(str, mnk))
            mnksig = ("LIBXSMM_LD(" + mstr + ", " + nstr + "), " +
                      "LIBXSMM_LD(" + nstr + ", " + mstr + "), " + kstr)
            ldxsig = ("LIBXSMM_LD(" + mstr + ", " + nstr + "), " +
                      kstr + ", LIBXSMM_LD(" + mstr + ", " + nstr + ")")
            # prefer registering double-precision kernels
            # when approaching an exhausted registry
            if (2 != precision):  # only single-precision
                print("if (LIBXSMM_GEMM_NO_BYPASS_DIMS(" + mnksig + ") &&")
                print("    LIBXSMM_GEMM_NO_BYPASS_DIMS(" + ldxsig + ")) {")
                print("  LIBXSMM_GEMM_DESCRIPTOR(desc, LIBXSMM_ALIGNMENT, " +
                      "LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_F32PREC,")
                print("    " + mnksig + ", " + ldxsig + ",")
                print("    LIBXSMM_ALPHA, LIBXSMM_BETA, INTERNAL_PREFETCH);")
                print("  LIBXSMM_HASH_FUNCTION_CALL(hash, indx, &desc);" +
                      " func.smm = (libxsmm_smmfunction)libxsmm_smm_" +
                      mnkstr + ";")
                print("  internal_register_static_code(" +
                      "&desc, indx, hash, func, result);")
                print("}")
        print("#if defined(_MSC_VER)")
        print("# pragma warning(pop)")
        print("#endif")
    elif (1 < argc):
        print("/* no static code */")
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
