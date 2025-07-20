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
import pathlib
import ford


def on_pre_build(config):
    directory = config.docs_dir if config else "documentation"
    proj_file = "libxsmm_fortran.md"

    with open(f"{directory}/libxsmm_fortran.md", "r") as project_file:
        proj_docs = project_file.read()
        proj_docs, proj_data = ford.load_settings(proj_docs, directory, proj_file)
        proj_data, proj_docs = ford.parse_arguments({}, proj_docs, proj_data, directory)

        if proj_data and proj_docs:
            ford.main(proj_data, proj_docs)


if __name__ == "__main__":
    on_pre_build(None)
