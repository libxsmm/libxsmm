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
from string import Template
import libxsmm_utilities
import fnmatch
import sys


if __name__ == "__main__":
    argc = len(sys.argv)
    if (1 < argc):
        # required argument(s)
        filename = sys.argv[1]

        # default configuration if no arguments are given
        ilp64 = offload = flags = threshold = 0
        sync = jit = 1
        alpha = beta = 1
        alignment = 64
        prefetch = -1
        wrap = 2
        mnklist = list()

        # optional argument(s)
        if (2 < argc):
            ilp64 = int(sys.argv[2])
        if (3 < argc):
            offload = int(sys.argv[3])
        if (4 < argc):
            alignment = libxsmm_utilities.sanitize_alignment(int(sys.argv[4]))
        if (5 < argc):
            prefetch = int(sys.argv[5])
        if (6 < argc):
            threshold = int(sys.argv[6])
        if (7 < argc):
            sync = int(sys.argv[7])
        if (8 < argc):
            jit = int(sys.argv[8])
        if (9 < argc):
            flags = int(sys.argv[9])
        if (10 < argc):
            alpha = int(sys.argv[10])
        if (11 < argc):
            beta = int(sys.argv[11])
        if (12 < argc):
            wrap = int(sys.argv[12])
        if (13 < argc):
            mnklist = sorted(libxsmm_utilities.load_mnklist(sys.argv[13:], 0))

        version, branch = \
            libxsmm_utilities.version_branch()
        major, minor, update, patch = \
            libxsmm_utilities.version_numbers(version)

        if (0 == threshold):
            threshold = 128 * 128 * 128
        maxmnk = libxsmm_utilities.max_mnk(mnklist, threshold)
        maxdim = int(maxmnk ** (1.0 / 3.0) + 0.5)
        avgdim = int(0.5 * maxdim + 0.5)

        avgm = libxsmm_utilities.median(
            list(map(lambda mnk: mnk[0], mnklist)), avgdim, False)
        avgn = libxsmm_utilities.median(
            list(map(lambda mnk: mnk[1], mnklist)), avgdim, False)
        avgk = libxsmm_utilities.median(
            list(map(lambda mnk: mnk[2], mnklist)), avgdim, False)

        maxm = libxsmm_utilities.max_mnk(mnklist, avgdim, 0)
        maxn = libxsmm_utilities.max_mnk(mnklist, avgdim, 1)
        maxk = libxsmm_utilities.max_mnk(mnklist, avgdim, 2)

        substitute = {
            "VERSION":    version,
            "BRANCH":     branch,
            "MAJOR":      major,
            "MINOR":      minor,
            "UPDATE":     update,
            "PATCH":      patch,
            "ALIGNMENT":  alignment,
            "PREFETCH":   [-1, prefetch][0 <= prefetch],
            "MAX_MNK":    maxmnk,
            "MAX_M":      [maxdim, maxm][avgm < maxm],
            "MAX_N":      [maxdim, maxn][avgn < maxn],
            "MAX_K":      [maxdim, maxk][avgk < maxk],
            "AVG_M":      avgm,
            "AVG_N":      avgn,
            "AVG_K":      avgk,
            "FLAGS":      flags,
            "ILP64":      [0, 1][0 != ilp64],
            "ALPHA":      alpha,
            "BETA":       beta,
            "WRAP":       [1, 2][0 == (wrap % 2) or (0 > wrap)],
            "SYNC":       [0, 1][0 != sync],
            "JIT":        [0, 1][0 != jit],
            "LIBXSMM_OFFLOAD_BUILD":
                ["", "\n#define LIBXSMM_OFFLOAD_BUILD"][0 != offload],
            "MNK_INTERFACE_LIST": ""
        }

        template = Template(open(filename, "r").read())
        if (fnmatch.fnmatch(filename, "*.h*")):
            print(template.substitute(substitute))
        else:
            substitute["BLASINT_KIND"] = ["C_INT", "C_LONG_LONG"][0 != ilp64]
            print(template.safe_substitute(substitute))
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
