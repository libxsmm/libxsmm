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


def create_dispatch(typeflag, mnklist):
    print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) libxsmm_" + typeflag + "mm_function libxsmm_" + typeflag + "mm_dispatch(int m, int n, int k)"
    print "{"
    print "  static const libxsmm_" + typeflag + "mm_function functions[] = {"
    i, n, mnklen = 0, 6, len(mnklist)
    for mnk in mnklist:
        if (0 == ((i + 0) % n)): sys.stdout.write("    ")
        sys.stdout.write("libxsmm_" + typeflag + "mm_" + "_".join(map(str, mnk)))
        i = i + 1
        if (i < mnklen):
            sys.stdout.write([",\n", ", "][0 != (i % n)])
        else:
            print
    print "  };"
    print "  const int i = index(m, n, k);"
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
        print "  static const int indices[] = {"
        mnklist = libxsmm_utilities.load_mnklist(sys.argv)
        i, n, mnklen = 0, 30, len(mnklist)
        for mnk in mnklist:
            if (0 == ((i + 0) % n)): sys.stdout.write("    ")
            sys.stdout.write(", ".join(map(str, mnk)))
            i = i + 1
            if (i < mnklen):
                sys.stdout.write([",\n", ", "][0 != (i % n)])
            else:
                print
        print "  };"
        print "  const int mnk[] = { m, n, k }, *const hit = (const int*)bsearch(mnk, indices, " + str(len(mnklist)) + ", 3 * sizeof(*indices), compare);"
        print "  return 0 != hit ? ((hit - indices) / 3) : -1;"
        print "}"
        print
        print
        create_dispatch("s", mnklist)
        print
        print
        create_dispatch("d", mnklist)
        print
        print
        print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_init()"
        print "{"
        print "  libxsmm_smm_dispatch(0, 0, 0);"
        print "  libxsmm_dmm_dispatch(0, 0, 0);"
        print "}"
    else:
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
