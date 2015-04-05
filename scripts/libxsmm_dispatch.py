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


def create_dispatch_direct_function(typeflag, mnklist):
    print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) libxsmm_" + typeflag + "mm_function libxsmm_" + typeflag + "mm_dispatch(int m, int n, int k)"
    print "{"
    maxm = libxsmm_utilities.max_mnk(mnklist, 0, 0)
    maxn = libxsmm_utilities.max_mnk(mnklist, 0, 1)
    maxk = libxsmm_utilities.max_mnk(mnklist, 0, 2)
    d, h = maxk + 1, maxn + 1
    print "  static /*const*/ libxsmm_" + typeflag + "mm_function functions[/*" + str(d * h * (maxm + 1)) + "*/] = {"
    sys.stdout.write("    ")
    begin, m, n, r = 0, 0, 0, 8
    s = r * 6
    for mnk in mnklist:
        end = calc_direct_index(mnk, d, h)
        for i in range(begin, end):
            m = m + 1; n = n + 1
            if (0 == (m % s)):
                sys.stdout.write(",\n    ")
            elif (1 < n):
                sys.stdout.write(", ")
            sys.stdout.write("0")
        begin, m, n = end + 1, m + r, n + 1
        if (r > (m % s)):
            sys.stdout.write(",\n    ")
            m = m + m % s
        elif (1 < n):
            sys.stdout.write(", ")
        sys.stdout.write("libxsmm_" + typeflag + "mm_" + "_".join(map(str, mnk)))
    print
    print "  };"
    print "  return (" + str(maxm) + " >= m && " + str(maxn) + " >= n && " + str(maxk) + " >= k) " + \
                "? functions[" + str(d) + "*(" + str(h) + "*m+n)+k] " + \
                ": 0;"
    print "}"


def create_dispatch_direct(mnklist):
    print "#include <libxsmm.h>"
    print
    print
    create_dispatch_direct_function("s", mnklist)
    print
    print
    create_dispatch_direct_function("d", mnklist)


def create_dispatch_bsearch_function(typeflag, mnklist):
    print "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) libxsmm_" + typeflag + "mm_function libxsmm_" + typeflag + "mm_dispatch(int m, int n, int k)"
    print "{"
    i, n, mnklen = 0, 6, len(mnklist)
    print "  static /*const*/ libxsmm_" + typeflag + "mm_function functions[/*" + str(mnklen) + "*/] = {"
    sys.stdout.write("    ")
    for mnk in mnklist:
        i = i + 1
        sys.stdout.write("libxsmm_" + typeflag + "mm_" + "_".join(map(str, mnk)))
        if (i < mnklen):
            sys.stdout.write([", ", ",\n    "][0 == (i % n)])
    print
    print "  };"
    print "  const int i = libxsmm_dispatch_index(m, n, k);"
    print "  return 0 <= i ? functions[i] : 0;"
    print "}"


def create_dispatch_bsearch(mnklist):
    print "#include <libxsmm.h>"
    print
    print "#if defined(LIBXSMM_OFFLOAD)"
    print "# pragma offload_attribute(push,target(mic))"
    print "#endif"
    print "#include <stdlib.h>"
    print "#if defined(LIBXSMM_OFFLOAD)"
    print "# pragma offload_attribute(pop)"
    print "#endif"
    print
    print
    print "LIBXSMM_TARGET(mic) int libxsmm_dispatch_compare3(const void* a, const void* b)"
    print "{"
    print "  const int *const ia = (const int*)a, *const ib = (const int*)b;"
    print "  const int d0 = ia[0] - ib[0], d1 = ia[1] - ib[1], d2 = ia[2] - ib[2];"
    print "  return 0 != d0 ? d0 : (0 != d1 ? d1 : d2);"
    print "}"
    print
    print
    print "LIBXSMM_TARGET(mic) int libxsmm_dispatch_index(int m, int n, int k)"
    print "{"
    i, n, mnklen = 0, 12, len(mnklist)
    print "  static /*const*/ int indices[/*" + str(3 * mnklen) + "*/] = {"
    sys.stdout.write("    ")
    for mnk in mnklist:
        i = i + 1
        sys.stdout.write(", ".join(map(str, mnk)))
        if (i < mnklen):
            sys.stdout.write([", ", ",\n    "][0 == (i % n)])
    print
    print "  };"
    print "  const int* hit = 0;"
    print "  int mnk[3];"
    print
    print "  mnk[0] = m; mnk[1] = n; mnk[2] = k;"
    print "  hit = (const int*)bsearch(mnk, indices, " + str(mnklen) + ", 3 * sizeof(*indices), libxsmm_dispatch_compare3);"
    print "  return 0 != hit ? ((int)(hit - indices) / 3) : -1;"
    print "}"
    print
    print
    create_dispatch_bsearch_function("s", mnklist)
    print
    print
    create_dispatch_bsearch_function("d", mnklist)


if __name__ == '__main__':
    argc = len(sys.argv)
    if (3 < argc):
        threshold, sparsity = int(sys.argv[1]), int(sys.argv[2])
        mnklist = libxsmm_utilities.load_mnklist(sys.argv[3:], 0)
        maxmnk = libxsmm_utilities.max_mnk(mnklist, threshold)
        maxm = libxsmm_utilities.max_mnk(mnklist, 0, 0)
        maxn = libxsmm_utilities.max_mnk(mnklist, 0, 1)
        maxk = libxsmm_utilities.max_mnk(mnklist, 0, 2)
        maxsize = maxm * maxn * maxk
        if (maxsize <= (sparsity * maxmnk) and maxsize <= 65536):
            create_dispatch_direct(mnklist)
        else:
            create_dispatch_bsearch(mnklist)
    else:
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
