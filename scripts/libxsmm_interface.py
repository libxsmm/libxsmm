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
from string import Template
import libxsmm_utilities
import sys, fnmatch


if __name__ == "__main__":
    argc = len(sys.argv)
    if (1 < argc):
        # required argument(s)
        filename = sys.argv[1]

        # optional argument(s)
        row_major = int(sys.argv[2]) if (2 < argc) else 1
        alignment = libxsmm_utilities.sanitize_alignment(int(sys.argv[3])) if (3 < argc) else 64
        aligned_stores = libxsmm_utilities.sanitize_alignment(int(sys.argv[4])) if (4 < argc) else 1
        aligned_loads = libxsmm_utilities.sanitize_alignment(int(sys.argv[5])) if (5 < argc) else 1
        threshold = int(sys.argv[6]) if (6 < argc) else 0
        mnklist = libxsmm_utilities.load_mnklist(sys.argv[7:], 0, threshold) if (7 < argc) else list()

        template = Template(open(filename, "r").read())
        maxmnk = libxsmm_utilities.max_mnk(mnklist, threshold)
        avgdim = int(maxmnk ** (1.0 / 3.0) + 0.5)
        mlist = libxsmm_utilities.make_mlist(mnklist); mlist.append(avgdim)
        nlist = libxsmm_utilities.make_nlist(mnklist); nlist.append(avgdim)
        klist = libxsmm_utilities.make_klist(mnklist); klist.append(avgdim)

        substitute = { \
            "ALIGNMENT":      alignment, \
            "ALIGNED_STORES": aligned_stores, \
            "ALIGNED_LOADS":  aligned_loads, \
            "ALIGNED_MAX":    max(alignment, aligned_stores, aligned_loads), \
            "ROW_MAJOR":      1 if (0 != row_major) else 0, \
            "COL_MAJOR":      0 if (0 != row_major) else 1, \
            "MAX_MNK":        maxmnk, \
            "MAX_M":          libxsmm_utilities.max_mnk(mnklist, avgdim, 0), \
            "MAX_N":          libxsmm_utilities.max_mnk(mnklist, avgdim, 1), \
            "MAX_K":          libxsmm_utilities.max_mnk(mnklist, avgdim, 2), \
            "AVG_M":          libxsmm_utilities.median(mlist), \
            "AVG_N":          libxsmm_utilities.median(nlist), \
            "AVG_K":          libxsmm_utilities.median(klist), \
            "MNK_INTERFACE_LIST": "" \
        }

        if (fnmatch.fnmatch(filename, "*.h*")):
            for mnk in mnklist:
                mnkstr = "_".join(map(str, mnk))
                substitute["MNK_INTERFACE_LIST"] += "\n"
                substitute["MNK_INTERFACE_LIST"] += "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_smm_" + mnkstr + "(const float* a, const float* b, float* c);\n"
                substitute["MNK_INTERFACE_LIST"] += "LIBXSMM_EXTERN_C LIBXSMM_TARGET(mic) void libxsmm_dmm_" + mnkstr + "(const double* a, const double* b, double* c);\n"
            print template.substitute(substitute)
        else:
            print template.safe_substitute(substitute)
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
