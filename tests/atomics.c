/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_source.h>

#if !defined(ATOMIC_KIND)
# define ATOMIC_KIND LIBXSMM_ATOMIC_RELAXED
#endif


int main(void)
{
  LIBXSMM_ALIGNED(LIBXSMM_ATOMIC_LOCKTYPE lock = 0/*unlocked*/, LIBXSMM_ALIGNMENT);
  int result = EXIT_SUCCESS;
  int mh = 1051981, hp, tmp;

  LIBXSMM_NONATOMIC_STORE(&hp, 25071975, ATOMIC_KIND);
  tmp = LIBXSMM_NONATOMIC_LOAD(&hp, ATOMIC_KIND);
  if (tmp != LIBXSMM_ATOMIC_LOAD(&hp, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if (mh != LIBXSMM_NONATOMIC_SUB_FETCH(&hp, 24019994, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if (mh != LIBXSMM_ATOMIC_FETCH_ADD(&hp, 24019994, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  LIBXSMM_ATOMIC_STORE(&tmp, mh, ATOMIC_KIND);
  if (25071975 != LIBXSMM_NONATOMIC_FETCH_OR(&hp, tmp, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if ((25071975 | mh) != hp) {
    result = EXIT_FAILURE;
  }
  /* check if non-atomic and atomic are compatible */
  if (LIBXSMM_NONATOMIC_TRYLOCK(&lock, ATOMIC_KIND)) {
    if (LIBXSMM_ATOMIC_TRYLOCK(&lock, ATOMIC_KIND)) {
      result = EXIT_FAILURE;
    }
    LIBXSMM_NONATOMIC_RELEASE(&lock, ATOMIC_KIND);
    if (0 != lock) result = EXIT_FAILURE;
  }
  else {
    result = EXIT_FAILURE;
  }

  LIBXSMM_ATOMIC_ACQUIRE(&lock, LIBXSMM_SYNC_NPAUSE, ATOMIC_KIND);
  if (0 == lock) result = EXIT_FAILURE;
  if (LIBXSMM_ATOMIC_TRYLOCK(&lock, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if (LIBXSMM_ATOMIC_TRYLOCK(&lock, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if (0 == lock) result = EXIT_FAILURE;
  LIBXSMM_ATOMIC_RELEASE(&lock, ATOMIC_KIND);
  if (0 != lock) result = EXIT_FAILURE;

  return result;
}

