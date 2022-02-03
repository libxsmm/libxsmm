#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Alexander Heinecke (Intel Corp.)
###############################################################################

if [ $# -ne 1 ]
then
  echo "Usage: $(basename $0) mesh.nc"
  exit
fi

NCFILE=$1

ncdump -c ${NCFILE} | grep elements | head -n1 | awk '{print $3}' > ${NCFILE}.size
ncdump -v element_neighbors ${NCFILE} | sed -e '1,/data:/d' -e '$d' | grep "," | sed 's/,//g' | sed 's/;//g' | sed 's/  //g' > ${NCFILE}.neigh
ncdump -v element_boundaries ${NCFILE} | sed -e '1,/data:/d' -e '$d' | grep "," | sed 's/,//g' | sed 's/;//g' | sed 's/  //g' > ${NCFILE}.bound
ncdump -v element_neighbor_sides ${NCFILE} | sed -e '1,/data:/d' -e '$d' | grep "," | sed 's/,//g' | sed 's/;//g' | sed 's/  //g' > ${NCFILE}.sides
ncdump -v element_side_orientations ${NCFILE} | sed -e '1,/data:/d' -e '$d' | grep "," | sed 's/,//g' | sed 's/;//g' | sed 's/  //g' > ${NCFILE}.orient

