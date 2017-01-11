#!/usr/bin/env python
###############################################################################
## Copyright (c) 2014-2017, Intel Corporation                                ##
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
from string import Template
import libxsmm_utilities
import sys, fnmatch


if __name__ == "__main__":
    argc = len(sys.argv)
    if (1 < argc):
        # required argument(s)
        filename = sys.argv[1]

        # optional argument(s)
        if (2 < argc): ilp64 = int(sys.argv[2])
        else: ilp64 = 0
        if (3 < argc): big = int(sys.argv[3])
        else: big = 0
        if (4 < argc): offload = int(sys.argv[4])
        else: offload = 0
        if (5 < argc): alignment = libxsmm_utilities.sanitize_alignment(int(sys.argv[5]))
        else: alignment = 64
        if (6 < argc): prefetch = int(sys.argv[6])
        else: prefetch = 0
        if (7 < argc): threshold = int(sys.argv[7])
        else: threshold = 0
        if (8 < argc): sync = int(sys.argv[8])
        else: sync = 0
        if (9 < argc): jit = int(sys.argv[9])
        else: jit = 0
        if (10 < argc): flags = int(sys.argv[10])
        else: flags = 0
        if (11 < argc): alpha = int(sys.argv[11])
        else: alpha = 1
        if (12 < argc): beta = int(sys.argv[12])
        else: beta = 1
        if (13 < argc): mnklist = sorted(libxsmm_utilities.load_mnklist(sys.argv[13:], 0))
        else: mnklist = list()

        if (0 == threshold): threshold = 128 * 128 * 128
        template = Template(open(filename, "r").read())
        maxmnk = libxsmm_utilities.max_mnk(mnklist, threshold)
        maxdim = int(maxmnk ** (1.0 / 3.0) + 0.5)
        avgdim = int(0.5 * maxdim + 0.5)

        avgm = libxsmm_utilities.median(list(map(lambda mnk: mnk[0], mnklist)), avgdim, False)
        avgn = libxsmm_utilities.median(list(map(lambda mnk: mnk[1], mnklist)), avgdim, False)
        avgk = libxsmm_utilities.median(list(map(lambda mnk: mnk[2], mnklist)), avgdim, False)

        maxm = libxsmm_utilities.max_mnk(mnklist, avgdim, 0)
        maxn = libxsmm_utilities.max_mnk(mnklist, avgdim, 1)
        maxk = libxsmm_utilities.max_mnk(mnklist, avgdim, 2)

        substitute = { \
            "LIBXSMM_OFFLOAD_BUILD": ["", "\n#define LIBXSMM_OFFLOAD_BUILD"][0!=offload], \
            "ALIGNMENT":  alignment, \
            "PREFETCH":   [-1, prefetch][0<=prefetch], \
            "MAX_MNK":    maxmnk, \
            "MAX_M":      [maxdim, maxm][avgm<maxm], \
            "MAX_N":      [maxdim, maxn][avgn<maxn], \
            "MAX_K":      [maxdim, maxk][avgk<maxk], \
            "AVG_M":      avgm, \
            "AVG_N":      avgn, \
            "AVG_K":      avgk, \
            "FLAGS":      flags, \
            "ILP64":      [0, 1][0!=ilp64], \
            "ALPHA":      alpha, \
            "BETA":       beta, \
            "SYNC":       [0, 1][0!=sync], \
            "JIT":        [0, 1][0!=jit], \
            "BIG":        big \
        }

        if (fnmatch.fnmatch(filename, "*.h*")):
            print(template.substitute(substitute))
        else:
            substitute["BLASINT_KIND"] = ["C_INT", "C_LONG_LONG"][0!=ilp64]
            print(template.safe_substitute(substitute))
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
