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
import sys
import os


def calc_direct_index(mnk, d, h):
    return d * (h * (mnk[0]) + mnk[1]) + mnk[2]


def create_dispatch(mnklist):
    print "#include <libxsmm_dispatch.h>"
    print "#include <libxsmm.h>"
    print
    print "#if defined(NDEBUG)"
    print "# define LIBXSMM_DISPATCH_CHECK(DISP) DISP"
    print "#else"
    print "# if defined(LIBXSMM_OFFLOAD_BUILD)"
    print "# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))"
    print "# endif"
    print "# include <assert.h>"
    print "# if defined(LIBXSMM_OFFLOAD_BUILD)"
    print "# pragma offload_attribute(pop)"
    print "# endif"
    print "# define LIBXSMM_DISPATCH_CHECK(DISP) assert(NULL == (DISP))"
    print "#endif"
    print
    print
    print "LIBXSMM_RETARGETABLE extern int libxsmm_init;"
    print
    print
    print "LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_initialize()"
    print "{"
    print "  if (0 == libxsmm_init) {"
    print "    struct { int m, n, k; } args;"
    for mnk in mnklist:
        mnkstr = "_".join(map(str, mnk))
        print "    args.m = " + str(mnk[0]) + "; args.n = " + str(mnk[1]) + "; args.k = " + str(mnk[2]) + ";"
        print "    LIBXSMM_DISPATCH_CHECK(libxsmm_dispatch(&args, sizeof(args), 0, (libxsmm_function)libxsmm_smm_" + mnkstr + "));"
        print "    LIBXSMM_DISPATCH_CHECK(libxsmm_dispatch(&args, sizeof(args), 1, (libxsmm_function)libxsmm_dmm_" + mnkstr + "));"
    print "    libxsmm_init = 1;"
    print "  }"
    print "}"


if __name__ == "__main__":
    argc = len(sys.argv)
    if (2 < argc):
        threshold = int(sys.argv[1])
        mnklist = libxsmm_utilities.load_mnklist(sys.argv[2:], 0, threshold)
        create_dispatch(mnklist)
    elif (1 < argc):
        print "#include <libxsmm.h>"
        print
        print
        print "LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmm_function libxsmm_dmm_dispatch(int m, int n, int k)"
        print "{"
        print "  return NULL;"
        print "}"
        print
        print
        print "LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smm_function libxsmm_smm_dispatch(int m, int n, int k)"
        print "{"
        print "  return NULL;"
        print "}"
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
