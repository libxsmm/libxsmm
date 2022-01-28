/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_EXT_H
#define LIBXSMM_EXT_H

#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(_OPENMP)
# if !defined(__INTEL_COMPILER)
#   if defined(__clang__)
#     pragma clang diagnostic push
#   elif defined(__GNUC__) && LIBXSMM_VERSION2(4, 6) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)
#     pragma GCC diagnostic push
#   endif
#   if defined(__clang__)
#     pragma clang diagnostic ignored "-Wpedantic"
#   elif defined(__GNUC__) && LIBXSMM_VERSION2(4, 6) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)
#     pragma GCC diagnostic ignored "-Wpedantic"
#   endif
# endif
# include <omp.h>
# if defined(LIBXSMM_TRACE_CALLERID_GCCBUILTIN) && !defined(__INTEL_COMPILER)
#   if defined(__clang__)
#     pragma clang diagnostic pop
#   elif defined(__GNUC__) && LIBXSMM_VERSION2(4, 6) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)
#     pragma GCC diagnostic pop
#   endif
# endif
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXSMM_EXT_H*/

