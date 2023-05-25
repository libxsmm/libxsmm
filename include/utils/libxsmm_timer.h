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
#ifndef LIBXSMM_UTILS_TIMER_H
#define LIBXSMM_UTILS_TIMER_H

#include "../libxsmm_typedefs.h"


LIBXSMM_EXTERN_C typedef struct libxsmm_timer_info {
  int tsc;
} libxsmm_timer_info;


/** Query timer properties. */
LIBXSMM_API int libxsmm_get_timer_info(libxsmm_timer_info* info);

/**
 * Returns the current clock tick of a monotonic timer source with
 * platform-specific resolution (not necessarily CPU cycles).
 */
LIBXSMM_API libxsmm_timer_tickint libxsmm_timer_tick(void);

/** Returns the difference between two timer ticks (cycles); avoids potential side-effects/assumptions of LIBXSMM_DIFF. */
LIBXSMM_API_INLINE libxsmm_timer_tickint libxsmm_timer_ncycles(libxsmm_timer_tickint tick0, libxsmm_timer_tickint tick1) {
  return LIBXSMM_DELTA(tick0, tick1);
}

/** Returns the duration (in seconds) between two values received by libxsmm_timer_tick. */
LIBXSMM_API double libxsmm_timer_duration(libxsmm_timer_tickint tick0, libxsmm_timer_tickint tick1);

#endif /*LIBXSMM_UTILS_TIMER_H*/
