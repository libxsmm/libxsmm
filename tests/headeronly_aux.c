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

/* must match definitions in headeronly.c */
#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif


LIBXSMM_EXTERN_C LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE) mmdispatch(int m, int n, int k);
LIBXSMM_EXTERN_C LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE) mmdispatch(int m, int n, int k)
{
  LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE) result;
#if defined(__cplusplus) /* C++ by chance: test libxsmm_mmfunction<> wrapper */
  const libxsmm_mmfunction<ITYPE, OTYPE> mmfunction(m, n, k);
  result = mmfunction.kernel().LIBXSMM_TPREFIX2(ITYPE, OTYPE, mm);
#else
  result = LIBXSMM_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/,
    NULL/*alpha*/, NULL/*beta*/,
    NULL/*flags*/, NULL/*prefetch*/);
#endif
  return result;
}

