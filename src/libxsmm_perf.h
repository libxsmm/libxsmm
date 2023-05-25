/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Maciej Debski (Google Inc.)
******************************************************************************/
#ifndef LIBXSMM_PERF_H
#define LIBXSMM_PERF_H

#include <libxsmm_typedefs.h>


LIBXSMM_API_INTERN void libxsmm_perf_init(libxsmm_timer_tickint (*timer_tick)(void));
LIBXSMM_API_INTERN void libxsmm_perf_finalize(void);
LIBXSMM_API_INTERN void libxsmm_perf_dump_code(
  const void* memory, size_t size,
  const char* name);

#endif /* LIBXSMM_PERF_H */
