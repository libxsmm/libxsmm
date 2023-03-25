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
import libxsmm_utilities
import fnmatch
import sys
import re


if __name__ == "__main__":
    argc = len(sys.argv)
    if 1 < argc:
        # required argument(s)
        filename = sys.argv[1]

        # default configuration if no arguments are given
        precision = 0  # all
        ifversion = 1  # interface
        prefetch = -1  # auto
        mnklist = list()

        # optional argument(s)
        if 2 < argc:
            ivalue = int(sys.argv[2])
            ifversion = ivalue >> 2
            precision = ivalue & 3
        if 3 < argc:
            prefetch = int(sys.argv[3])
        if 4 < argc:
            mnklist = sorted(libxsmm_utilities.load_mnklist(sys.argv[4:], 0))

        template = Template(open(filename, "r").read())
        if fnmatch.fnmatch(filename, "*.h"):
            optional = [", ...", ""][0 <= prefetch]
            substitute = {"MNK_INTERFACE_LIST": ""}
            for mnk in mnklist:
                mnkstr = "_".join(map(str, mnk))
                if 2 != precision:
                    pfsig = [
                        optional + ");",
                        ",\n  "
                        "const float* pa, "
                        "const float* pb, "
                        "const float* pc);",
                    ][0 < prefetch]
                    substitute["MNK_INTERFACE_LIST"] += (
                        "\nLIBXSMM_API void libxsmm_smm_"
                        + mnkstr
                        + "(const float* a, const float* b, float* c"
                        + pfsig
                    )
                if 1 != precision:
                    pfsig = [
                        optional + ");",
                        ",\n  "
                        "const double* pa, "
                        "const double* pb, "
                        "const double* pc);",
                    ][0 < prefetch]
                    substitute["MNK_INTERFACE_LIST"] += (
                        "\nLIBXSMM_API void libxsmm_dmm_"
                        + mnkstr
                        + "(const double* a, const double* b, double* c"
                        + pfsig
                    )
                if 0 == precision:
                    substitute["MNK_INTERFACE_LIST"] += "\n"
            if mnklist and 0 != precision:
                substitute["MNK_INTERFACE_LIST"] += "\n"
        else:  # Fortran interface
            if 1 > ifversion and 0 != ifversion:
                raise ValueError("Fortran interface level is inconsistent!")
            # Fortran's OPTIONAL allows to always generate an interface
            # with prefetch signature (more flexible usage)
            if 0 == prefetch:
                prefetch = -1
            version, branch, realversion = libxsmm_utilities.version_branch(16)
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
                "MNK_INTERFACE_LIST": "",
                "CONTIGUOUS": ["", ", CONTIGUOUS"][1 < ifversion],
            }
            if mnklist:
                substitute["MNK_INTERFACE_LIST"] += "\n        INTERFACE"
                optional = [", OPTIONAL", ""][0 < prefetch]
                bindc = ["", "BIND(C)"][0 < prefetch]
                for mnk in mnklist:
                    mnkstr = "_".join(map(str, mnk))
                    if 2 != precision:
                        pfsiga = [
                            ") BIND(C)\n",
                            ","
                            + "&".rjust(26 - len(mnkstr))
                            + "\n     &    pa, pb, pc) "
                            + bindc
                            + "\n",
                        ][0 != prefetch]
                        pfsigb = [
                            "",
                            "            REAL(C_FLOAT), "
                            "INTENT(IN)" + optional + " :: "
                            "pa(*), "
                            "pb(*), "
                            "pc(*)\n",
                        ][0 != prefetch]
                        substitute["MNK_INTERFACE_LIST"] += (
                            "\n          "
                            "PURE SUBROUTINE libxsmm_smm_"
                            + mnkstr
                            + "(a, b, c"
                            + pfsiga
                            + "            IMPORT :: C_FLOAT\n"
                            "            REAL(C_FLOAT), "
                            "INTENT(IN) :: a(*), b(*)\n"
                            "            REAL(C_FLOAT), "
                            "INTENT(INOUT) :: c(*)\n"
                            + pfsigb
                            + "          END SUBROUTINE"
                        )
                    if 1 != precision:
                        pfsiga = [
                            ") BIND(C)\n",
                            ","
                            + "&".rjust(26 - len(mnkstr))
                            + "\n     &    pa, pb, pc) "
                            + bindc
                            + "\n",
                        ][0 != prefetch]
                        pfsigb = [
                            "",
                            "            REAL(C_DOUBLE), "
                            "INTENT(IN)" + optional + " :: "
                            "pa(*), "
                            "pb(*), "
                            "pc(*)\n",
                        ][0 != prefetch]
                        substitute["MNK_INTERFACE_LIST"] += (
                            "\n          "
                            "PURE SUBROUTINE libxsmm_dmm_"
                            + mnkstr
                            + "(a, b, c"
                            + pfsiga
                            + "            IMPORT :: C_DOUBLE\n"
                            "            REAL(C_DOUBLE), "
                            "INTENT(IN) :: a(*), b(*)\n"
                            "            REAL(C_DOUBLE), "
                            "INTENT(INOUT) :: c(*)\n"
                            + pfsigb
                            + "          END SUBROUTINE"
                        )
                substitute["MNK_INTERFACE_LIST"] += "\n        END INTERFACE\n"
        txt = template.safe_substitute(substitute)
        sys.stdout.write(  # print without trailing newline
            re.sub(r"[ \t]+\r*\n", r"\n", txt)
        )
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
