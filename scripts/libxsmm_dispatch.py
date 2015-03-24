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
import sys


def create_dispatch(typeflag, mnklist):
    print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) libxsmm_" + typeflag + "mm_function libxsmm_" + typeflag + "mm_dispatch(int m, int n, int k)"
    print "{"
    print "  static const libxsmm_" + typeflag + "mm_function functions[] = {"
    i, n, mnklen = 0, 5, len(mnklist)
    for mnk in mnklist:
        if (0 == ((i + 0) % n)): sys.stdout.write("    ")
        sys.stdout.write("libxsmm_" + typeflag + "mm_" + str(mnk[0]) + "_" + str(mnk[1]) + "_" + str(mnk[2]))
        i = i + 1
        if (i < mnklen):
            sys.stdout.write([",\n", ", "][0 != (i % n)])
        else:
            print
    print "  };"
    print "  const int i = index(m, n, k);"
    print
    print "  return 0 <= i ? functions[i] : 0;"
    print "}"


if __name__ == '__main__':
    if (3 < len(sys.argv)):
        print "#include <libxsmm.h>"
        print
        print "#if defined(LIBXSMM_OFFLOAD)"
        print "# pragma offload_attribute(push,target(mic))"
        print "# include <stdlib.h>"
        print "# pragma offload_attribute(pop)"
        print "#else"
        print "# include <stdlib.h>"
        print "#endif"
        print
        print
        print "LIBXSMM_TARGET(mic) int compare(const void* a, const void* b)"
        print "{"
        print "  const int *const ia = (const int*)a, *const ib = (const int*)b;"
        print "  const int d0 = ia[0] - ib[0], d1 = ia[1] - ib[1], d2 = ia[2] - ib[2];"
        print "  return 0 != d0 ? d0 : (0 != d1 ? d1 : d2);"
        print "}"
        print
        print
        print "LIBXSMM_TARGET(mic) int index(int m, int n, int k)"
        print "{"
        mnklist = libxsmm_utilities.load_mnklist(sys.argv)
        print "  static const int indices[] = { " + ", ".join(  map(lambda mnk: ", ".join(map(str, mnk)), mnklist)).strip("[]") + " };"
        print "  const int mnk[] = { m, n, k };"
        print "  int i = 0;"
        print
        print "  return (LIBXSMM_MAX_MNK >= (m * n * k) && 0 <= (i = ((const int*)bsearch("
        print "      mnk, indices, " + str(len(mnklist)) + ", 3 * sizeof(*indices), compare)) - indices))"
        print "    ? (i / 3) : -1;"
        print "}"
        print
        print
        create_dispatch("s", mnklist)
        print
        print
        create_dispatch("d", mnklist)
    else:
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
