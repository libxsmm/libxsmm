#!/usr/bin/env python3
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################
import sys


if __name__ == "__main__":
    argc = len(sys.argv)
    if 6 == argc:
        precision = int(sys.argv[1])
        m, n, k = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        prefetch = int(sys.argv[5])

        mnkstr = str(m) + "_" + str(n) + "_" + str(k)
        optional = ["", ", ..."][0 > prefetch]
        signature = ["a, b, c", "a, b, c, pa, pb, pc"][0 < prefetch]
        if 2 != precision:
            pfsig = [
                optional + ")",
                "\n"
                ", const float* pa"
                ", const float* pb"
                ", const float* pc)",
            ][0 < prefetch]
            print
            print
            print(
                "LIBXSMM_API void libxsmm_smm_"
                + mnkstr
                + "(const float* a, const float* b, float* c"
                + pfsig
            )
            print("{")
            print(
                "#if defined(__AVX512F__) && "
                "defined(LIBXSMM_GENTARGET_skx_sp) && \\"
            )
            print("  !(defined(__AVX512PF__) && defined(__AVX512ER__))")
            print("  libxsmm_smm_" + mnkstr + "_skx(" + signature + ");")
            print(
                "#elif defined(__AVX512F__) && "
                "defined(LIBXSMM_GENTARGET_knl_sp)"
            )
            print("  libxsmm_smm_" + mnkstr + "_knl(" + signature + ");")
            print(
                "#elif defined(__AVX2__) && "
                "defined(LIBXSMM_GENTARGET_hsw_sp)"
            )
            print("  libxsmm_smm_" + mnkstr + "_hsw(" + signature + ");")
            print(
                "#elif defined(__AVX__) && "
                "defined(LIBXSMM_GENTARGET_snb_sp)"
            )
            print("  libxsmm_smm_" + mnkstr + "_snb(" + signature + ");")
            print(
                "#elif defined(__SSE3__) && "
                "defined(LIBXSMM_GENTARGET_wsm_sp)"
            )
            print("  libxsmm_smm_" + mnkstr + "_wsm(" + signature + ");")
            print("#else")
            print(
                "  const char transa = (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & "
                "LIBXSMM_FLAGS) ? 'N' : 'T');"
            )
            print(
                "  const char transb = (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & "
                "LIBXSMM_FLAGS) ? 'N' : 'T');"
            )
            print("  const float alpha = LIBXSMM_ALPHA, beta = LIBXSMM_BETA;")
            print(
                "  const libxsmm_blasint "
                "m = " + str(m) + ", "
                "n = " + str(n) + ", "
                "k = " + str(k) + ";"
            )
            if 0 < prefetch:
                print(
                    "  LIBXSMM_UNUSED(pa);"
                    " LIBXSMM_UNUSED(pb);"
                    " LIBXSMM_UNUSED(pc);"
                )
            print("#pragma GCC diagnostic push")
            print("#pragma GCC diagnostic ignored \"-Wcast-qual\"")
            print(
                "  LIBXSMM_INLINE_XGEMM(float, float, &transa, &transb,"
                " &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);"
            )
            print("#pragma GCC diagnostic pop")
            print("#endif")
            print("}")
            print
            print
            print(
                "LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_smm_"
                + mnkstr
                + ")(const float* a, const float* b, float* c"
                + pfsig
                + ";"
            )
            print(
                "LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_smm_"
                + mnkstr
                + ")(const float* a, const float* b, float* c"
                + pfsig
            )
            print("{")
            print("  libxsmm_smm_" + mnkstr + "(" + signature + ");")
            print("}")
        if 1 != precision:
            pfsig = [
                optional + ")",
                "\n"
                ", const double* pa"
                ", const double* pb"
                ", const double* pc)",
            ][0 < prefetch]
            print
            print
            print(
                "LIBXSMM_API void libxsmm_dmm_"
                + mnkstr
                + "(const double* a, const double* b, double* c"
                + pfsig
            )
            print("{")
            print(
                "#if defined(__AVX512F__) && "
                "defined(LIBXSMM_GENTARGET_skx_dp) && \\"
            )
            print("  !(defined(__AVX512PF__) && defined(__AVX512ER__))")
            print("  libxsmm_dmm_" + mnkstr + "_skx(" + signature + ");")
            print(
                "#elif defined(__AVX512F__) && "
                "defined(LIBXSMM_GENTARGET_knl_dp)"
            )
            print("  libxsmm_dmm_" + mnkstr + "_knl(" + signature + ");")
            print(
                "#elif defined(__AVX2__) && "
                "defined(LIBXSMM_GENTARGET_hsw_dp)"
            )
            print("  libxsmm_dmm_" + mnkstr + "_hsw(" + signature + ");")
            print(
                "#elif defined(__AVX__) && "
                "defined(LIBXSMM_GENTARGET_snb_dp)"
            )
            print("  libxsmm_dmm_" + mnkstr + "_snb(" + signature + ");")
            print(
                "#elif defined(__SSE3__) && "
                "defined(LIBXSMM_GENTARGET_wsm_dp)"
            )
            print("  libxsmm_dmm_" + mnkstr + "_wsm(" + signature + ");")
            print("#else")
            print(
                "  const char transa = (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & "
                "LIBXSMM_FLAGS) ? 'N' : 'T');"
            )
            print(
                "  const char transb = (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & "
                "LIBXSMM_FLAGS) ? 'N' : 'T');"
            )
            print("  const double alpha = LIBXSMM_ALPHA, beta = LIBXSMM_BETA;")
            print(
                "  const libxsmm_blasint "
                "m = " + str(m) + ", "
                "n = " + str(n) + ", "
                "k = " + str(k) + ";"
            )
            if 0 < prefetch:
                print(
                    "  LIBXSMM_UNUSED(pa);"
                    " LIBXSMM_UNUSED(pb);"
                    " LIBXSMM_UNUSED(pc);"
                )
            print("#pragma GCC diagnostic push")
            print("#pragma GCC diagnostic ignored \"-Wcast-qual\"")
            print(
                "  LIBXSMM_INLINE_XGEMM(double, double, &transa, &transb,"
                " &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);"
            )
            print("#pragma GCC diagnostic pop")
            print("#endif")
            print("}")
            print
            print
            print(
                "LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dmm_"
                + mnkstr
                + ")(const double* a, const double* b, double* c"
                + pfsig
                + ";"
            )
            print(
                "LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dmm_"
                + mnkstr
                + ")(const double* a, const double* b, double* c"
                + pfsig
            )
            print("{")
            print("  libxsmm_dmm_" + mnkstr + "(" + signature + ");")
            print("}")
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
