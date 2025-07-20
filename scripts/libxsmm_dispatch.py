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
import libxsmm_utilities
import sys
import os


if __name__ == "__main__":
    argc = len(sys.argv)
    if 1 < argc:
        arg1_filename = [sys.argv[1], ""]["0" == sys.argv[1]]
        arg1_isfile = os.path.isfile(arg1_filename)
        base = 1
        if arg1_isfile:
            print("#if !defined(_WIN32)")
            print("{ static const char *const build_state =")
            print('#   include "../' + os.path.basename(arg1_filename) + '"')
            print("  ;")
            print("  internal_build_state = build_state;")
            print("}")
            print("#endif")
            base = 2
        if (base + 2) < argc:
            precision = int(sys.argv[base + 0])
            threshold = int(sys.argv[base + 1])
            mnklist = libxsmm_utilities.load_mnklist(
                sys.argv[base + 2 :], 0  # noqa: E203
            )
            print(
                "/* omit registering code if JIT is enabled"
                " and if an ISA extension is found"
            )
            print(
                " * which is beyond the static code"
                " path used to compile the library"
            )
            print(" */")
            print("#if (0 != LIBXSMM_JIT) && !defined(__MIC__)")
            print(
                "if (LIBXSMM_X86_GENERIC > libxsmm_target_archid "
                "/* JIT code gen. is not available */"
            )
            print(
                "   /* conditions allows to avoid JIT "
                "(if static code is good enough) */"
            )
            print(
                "   || (LIBXSMM_STATIC_TARGET_ARCH == libxsmm_target_archid)"
            )
            print("   || (LIBXSMM_X86_AVX512_SKX <= libxsmm_target_archid &&")
            print("       libxsmm_cpuid_vlen32(LIBXSMM_STATIC_TARGET_ARCH) ==")
            print("       libxsmm_cpuid_vlen32(libxsmm_target_archid)))")
            print("#endif")
            print("{")
            print("  libxsmm_xmmfunction func;")
            for mnk in mnklist:
                mstr, nstr, kstr, mnkstr = (
                    str(mnk[0]),
                    str(mnk[1]),
                    str(mnk[2]),
                    "_".join(map(str, mnk)),
                )
                mnksig = mstr + ", " + nstr + ", " + kstr
                # prefer registering double-precision kernels
                # when approaching an exhausted registry
                if 1 != precision:  # only double-precision
                    print(
                        "  func.dmm = (libxsmm_dmmfunction)libxsmm_dmm_"
                        + mnkstr
                        + ";"
                    )
                    print(
                        "  internal_register_static_code("
                        + "LIBXSMM_DATATYPE_F64, "
                        + mnksig
                        + ", func, new_registry);"
                    )
            for mnk in mnklist:
                mstr, nstr, kstr, mnkstr = (
                    str(mnk[0]),
                    str(mnk[1]),
                    str(mnk[2]),
                    "_".join(map(str, mnk)),
                )
                mnksig = mstr + ", " + nstr + ", " + kstr
                # prefer registering double-precision kernels
                # when approaching an exhausted registry
                if 2 != precision:  # only single-precision
                    print(
                        "  func.smm = (libxsmm_smmfunction)libxsmm_smm_"
                        + mnkstr
                        + ";"
                    )
                    print(
                        "  internal_register_static_code("
                        + "LIBXSMM_DATATYPE_F32, "
                        + mnksig
                        + ", func, new_registry);"
                    )
            print("}")
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
