#!/usr/bin/env bash
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

GIT=$(command -v git)

if [ "${GIT}" ]; then
  ${GIT} gc
  ${GIT} fsck --full
  ${GIT} reflog expire --expire=now --all
  # ${GIT} gc --prune=now
  ${GIT} gc --aggressive
  ${GIT} remote update --prune
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

