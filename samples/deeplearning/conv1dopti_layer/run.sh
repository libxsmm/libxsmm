###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Narendra Chaudhary (Intel Corp.)
###############################################################################


#!/bin/bash

export KMP_AFFINITY=compact,1,0,granularity=fine              # Set KMP affinity
# export KMP_BLOCKTIME=1

export OMP_NUM_THREADS=28                                     # Set number of threads
export LD_LIBRARY_PATH=../../../../libxsmm/lib/        # Set LD_LIBRARY_PATH

# python torch_example.py                                       # Run the pytorch example
python Efficiency_test.py                                       # Run the Efficiency test