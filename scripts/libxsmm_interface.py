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
        ilp64 = int(sys.argv[2]) if (2 < argc) else 0
        alignment = libxsmm_utilities.sanitize_alignment(int(sys.argv[3])) if (3 < argc) else 64
        row_major = int(sys.argv[4]) if (4 < argc) else 0
        prefetch = int(sys.argv[5]) if (5 < argc) else 0
        threshold = int(sys.argv[6]) if (6 < argc) else 0
        jit = int(sys.argv[7]) if (7 < argc) else 0
        flags = int(sys.argv[8]) if (8 < argc) else 0
        alpha = int(sys.argv[9]) if (9 < argc) else 1
        beta = int(sys.argv[10]) if (10 < argc) else 1
        mnklist = libxsmm_utilities.load_mnklist(sys.argv[11:], threshold) if (11 < argc) else list()

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

        version = libxsmm_utilities.version()
        major, minor, patch = libxsmm_utilities.version_number(version)

        substitute = { \
            "VERSION":    version, \
            "MAJOR":      major, \
            "MINOR":      minor, \
            "PATCH":      patch, \
            "ALIGNMENT":  alignment, \
            "ROW_MAJOR":  1 if (0 != row_major) else 0, \
            "COL_MAJOR":  0 if (0 != row_major) else 1, \
            "PREFETCH":   prefetch, \
            "MAX_MNK":    maxmnk, \
            "MAX_M":      maxm if (avgm < maxm) else maxdim, \
            "MAX_N":      maxn if (avgn < maxn) else maxdim, \
            "MAX_K":      maxk if (avgk < maxk) else maxdim, \
            "AVG_M":      avgm, \
            "AVG_N":      avgn, \
            "AVG_K":      avgk, \
            "FLAGS":      flags, \
            "ILP64":      1 if (0 != ilp64) else 0, \
            "ALPHA":      alpha, \
            "BETA":       beta, \
            "JIT":        1 if (0 != jit) else 0, \
            "MNK_INTERFACE_LIST": "" \
        }

        if (fnmatch.fnmatch(filename, "*.h*")):
            for mnk in mnklist:
                mnkstr = "_".join(map(str, mnk))
                substitute["MNK_INTERFACE_LIST"] += "\n" \
                    "LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_smm_" + mnkstr + "(\n" + \
                      "  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c" + \
                      (",\n  const float* pa, const float* pb, const float* pc);\n" if (0 != prefetch) else ");\n") + \
                    "LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_dmm_" + mnkstr + "(\n" + \
                      "  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c" + \
                      (",\n  const double* pa, const double* pb, const double* pc);\n" if (0 != prefetch) else ");\n")
            print(template.substitute(substitute))
        else:
            if (mnklist):
                substitute["MNK_INTERFACE_LIST"] += "\n"
                for mnk in mnklist:
                    mnkstr = "_".join(map(str, mnk))
                    substitute["MNK_INTERFACE_LIST"] += "\n        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm_" + mnkstr + ", libxsmm_dmm_" + mnkstr
                substitute["MNK_INTERFACE_LIST"] += "\n        INTERFACE"
                for mnk in mnklist:
                    mnkstr = "_".join(map(str, mnk))
                    substitute["MNK_INTERFACE_LIST"] += "\n" \
                        "          PURE SUBROUTINE libxsmm_smm_" + mnkstr + "(a, b, c" + \
                        ("," + "&".rjust(26 - len(mnkstr)) + "\n     &    pa, pb, pc) BIND(C)\n" if (0 != prefetch) else ") BIND(C)\n") + \
                        "            IMPORT :: C_FLOAT\n" \
                        "            REAL(C_FLOAT), INTENT(IN) :: a(*), b(*)\n" \
                        "            REAL(C_FLOAT), INTENT(INOUT) :: c(*)\n" + \
                        ("            REAL(C_FLOAT), INTENT(IN) :: pa(*), pb(*), pc(*)\n" if (0 != prefetch) else "") + \
                        "          END SUBROUTINE" \
                        "\n" \
                        "          PURE SUBROUTINE libxsmm_dmm_" + mnkstr + "(a, b, c" + \
                        ("," + "&".rjust(26 - len(mnkstr)) + "\n     &    pa, pb, pc) BIND(C)\n" if (0 != prefetch) else ") BIND(C)\n") + \
                        "            IMPORT :: C_DOUBLE\n" \
                        "            REAL(C_DOUBLE), INTENT(IN) :: a(*), b(*)\n" \
                        "            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)\n" + \
                        ("            REAL(C_DOUBLE), INTENT(IN) :: pa(*), pb(*), pc(*)\n" if (0 != prefetch) else "") + \
                        "          END SUBROUTINE"
                substitute["MNK_INTERFACE_LIST"] += "\n        END INTERFACE"
            print(template.safe_substitute(substitute))
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
