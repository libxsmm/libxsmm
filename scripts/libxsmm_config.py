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
from string import Template
from datetime import date
import libxsmm_utilities
import fnmatch
import sys


if __name__ == "__main__":
    argc = len(sys.argv)
    if 1 < argc:
        # required argument(s)
        filename = sys.argv[1]

        # default configuration if no arguments are given
        ilp64 = precision = flags = 0
        sync = jit = 1
        alpha = beta = 1
        cacheline = 64
        prefetch = -1
        wrap = 1
        malloc = 0
        mnklist = list()

        # optional argument(s)
        if 2 < argc:
            ilp64 = int(sys.argv[2])
        if 3 < argc:
            cacheline = libxsmm_utilities.sanitize_alignment(int(sys.argv[3]))
        if 4 < argc:
            precision = int(sys.argv[4])
        if 5 < argc:
            prefetch = int(sys.argv[5])
        if 6 < argc:
            sync = int(sys.argv[6])
        if 7 < argc:
            jit = int(sys.argv[7])
        if 8 < argc:
            flags = int(sys.argv[8])
        if 9 < argc:
            alpha = int(sys.argv[9])
        if 10 < argc:
            beta = int(sys.argv[10])
        if 11 < argc:
            malloc = int(sys.argv[11])
        if 12 < argc:
            mnklist = sorted(libxsmm_utilities.load_mnklist(sys.argv[12:], 0))

        version, branch, realversion = libxsmm_utilities.version_branch()
        major, minor, update, patch = libxsmm_utilities.version_numbers(
            version
        )

        substitute = {
            "VERSION": realversion,
            "BRANCH": branch,
            "MAJOR": major,
            "MINOR": minor,
            "UPDATE": update,
            "PATCH": patch,
            "DATE": date.today().strftime("%Y%m%d"),
            "CACHELINE": cacheline,
            "PREFETCH": [-1, prefetch][0 <= prefetch],
            "FLAGS": flags,
            "ILP64": [0, 1][0 != ilp64],
            "ALPHA": alpha,
            "BETA": beta,
            "MALLOC": malloc,
            "SYNC": [0, 1][0 != sync],
            "JIT": [0, 1][0 != jit],
            "MNK_PREPROCESSOR_LIST": "",
        }

        template = Template(open(filename, "r").read())
        if fnmatch.fnmatch(filename, "*.h"):
            if mnklist:
                first = mnklist[0]
            for mnk in mnklist:
                mnkstr = "_".join(map(str, mnk))
                if mnk != first:
                    substitute["MNK_PREPROCESSOR_LIST"] += "\n"
                if 2 != precision:
                    substitute["MNK_PREPROCESSOR_LIST"] += (
                        "# define LIBXSMM_SMM_" + mnkstr
                    )
                if mnk != first or 0 == precision:
                    substitute["MNK_PREPROCESSOR_LIST"] += "\n"
                if 1 != precision:
                    substitute["MNK_PREPROCESSOR_LIST"] += (
                        "# define LIBXSMM_DMM_" + mnkstr
                    )

            print(template.substitute(substitute))
        else:
            substitute["BLASINT_KIND"] = ["C_INT", "C_LONG_LONG"][0 != ilp64]
            print(template.safe_substitute(substitute))
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
