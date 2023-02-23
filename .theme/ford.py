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
import mkdocs.plugins
import ford
import io
import os


def on_serve(server, config, builder):
    proj_data, proj_docs, md = None, None, None

    with open(f"{config.docs_dir}/libxsmm_fortran.md", "r") as project_file:
        proj_docs = project_file.read()
        directory = os.path.dirname(project_file.name)
        proj_data, proj_docs, md = ford.parse_arguments(
            {}, proj_docs, directory
        )

    if proj_data and proj_docs and md:
        with ford.stdout_redirector(io.StringIO()):  # quiet
            ford.main(proj_data, proj_docs, md)

    return server
