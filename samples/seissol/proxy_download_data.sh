#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

WGET=$(command -v wget)

DATASET="LOH1_small merapi_15e5"
KINDS="bound neigh orient sides size"

for DATA in ${DATASET} ; do
  for KIND in ${KINDS} ; do
    ${WGET} -N https://github.com/hfp/libxsmm/raw/master/samples/seissol/${DATA}.nc.${KIND}
  done
done

