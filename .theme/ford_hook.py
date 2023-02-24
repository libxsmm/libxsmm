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
import mkdocs.plugins  # noqa: F401
import ford


def on_pre_build(config):
    docs_dir = config.docs_dir if config else "documentation"
    proj_data, proj_docs, md = None, None, None

    with open(f"{docs_dir}/libxsmm_fortran.md", "r") as project_file:
        proj_data, proj_docs, md = ford.parse_arguments(
            {}, project_file.read(), docs_dir
        )

    if proj_data and proj_docs and md:
        # with ford.stdout_redirector(io.StringIO()):  # quiet
        ford.main(proj_data, proj_docs, md)


if __name__ == "__main__":
    on_pre_build(None)
