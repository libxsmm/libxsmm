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
#include <libxsmm_source.h>

#if !defined(ELEMTYPE)
# define ELEMTYPE double
#endif


int main(void)
{
  int result = EXIT_SUCCESS;
  libxsmm_matdiff_info di[6], diff /*= { 0 }*/;
  /* http://www.netlib.org/lapack/lug/node75.html */
  const ELEMTYPE a[] = {
    (ELEMTYPE)1.00, (ELEMTYPE)2.00, (ELEMTYPE)3.00,
    (ELEMTYPE)4.00, (ELEMTYPE)5.00, (ELEMTYPE)6.00,
    (ELEMTYPE)7.00, (ELEMTYPE)8.00, (ELEMTYPE)10.0
  };
  const ELEMTYPE b[] = {
    (ELEMTYPE)0.44, (ELEMTYPE)2.36, (ELEMTYPE)3.04,
    (ELEMTYPE)3.09, (ELEMTYPE)5.87, (ELEMTYPE)6.66,
    (ELEMTYPE)7.36, (ELEMTYPE)7.77, (ELEMTYPE)9.07
  };
  const ELEMTYPE x[] = {
    (ELEMTYPE)1.00, (ELEMTYPE)100.0, (ELEMTYPE)9.00
  };
  const ELEMTYPE y[] = {
    (ELEMTYPE)1.10, (ELEMTYPE)99.00, (ELEMTYPE)11.0
  };
  const ELEMTYPE r[] = {
    (ELEMTYPE)0.00, (ELEMTYPE)0.00, (ELEMTYPE)0.00
  };
  const ELEMTYPE t[] = {
    (ELEMTYPE)0.01, (ELEMTYPE)0.02, (ELEMTYPE)0.01
  };

  /* no need to clear di; just the accumulator (diff) */
  libxsmm_matdiff_clear(&diff);

  result = libxsmm_matdiff(di + 0, LIBXSMM_DATATYPE(ELEMTYPE), 3/*m*/, 3/*n*/,
    a/*ref*/, b/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);

  if (EXIT_SUCCESS == result) {
    const double epsilon = libxsmm_matdiff_epsilon(di + 0);
    libxsmm_matdiff_reduce(&diff, di + 0);
    /* Epsilon (combined) */
    if (0.0000001 < LIBXSMM_ABS(epsilon - 0.1132714)) result = EXIT_FAILURE;
    /* One-norm */
    if (0.0000003 < LIBXSMM_ABS(di[0].norm1_abs - 1.8300000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[0].norm1_rel - 0.0963158)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000002 < LIBXSMM_ABS(di[0].normi_abs - 2.4400000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[0].normi_rel - 0.0976000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(di[0].normf_rel - 0.1074954)) result = EXIT_FAILURE;
    /* L2-norm */
    if (0.0000002 < LIBXSMM_ABS(di[0].l2_abs - 1.8742465)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[0].l2_rel - 0.6726295)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(di[0].l1_ref - 46.00)) result = EXIT_FAILURE;
    if (0.0000007 < LIBXSMM_ABS(di[0].l1_tst - 45.66)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000004 < LIBXSMM_ABS(di[0].linf_abs - 0.9300000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[0].linf_rel - 0.5600000)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(di[0].rsq - 0.9490077)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (2 != di[0].m || 2 != di[0].n) result = EXIT_FAILURE;
    if (a[3*di[0].m+di[0].n] != di[0].v_ref) result = EXIT_FAILURE;
    if (b[3*di[0].m+di[0].n] != di[0].v_tst) result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    result = libxsmm_matdiff(di + 1, LIBXSMM_DATATYPE(ELEMTYPE), 1/*m*/, 3/*n*/,
      x/*ref*/, y/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);
  }
  if (EXIT_SUCCESS == result) {
    const double epsilon = libxsmm_matdiff_epsilon(di + 1);
    libxsmm_matdiff_reduce(&diff, di + 1);
    /* Epsilon (combined) */
    if (0.0000001 < LIBXSMM_ABS(epsilon - 0.0223103)) result = EXIT_FAILURE;
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(di[1].norm1_abs - 3.1000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[1].norm1_rel - 0.0281818)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(di[1].normi_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[1].normi_rel - 0.0200000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(di[1].normf_rel - 0.0222918)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(di[1].l2_abs - 2.2383029)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[1].l2_rel - 0.2438908)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(di[1].l1_ref - 110.00)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[1].l1_tst - 111.10)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(di[1].linf_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[1].linf_rel - 0.2222222)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(di[1].rsq - 0.9991717)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (0 != di[1].m || 2 != di[1].n) result = EXIT_FAILURE;
    if (x[3*di[1].m+di[1].n] != di[1].v_ref) result = EXIT_FAILURE;
    if (y[3*di[1].m+di[1].n] != di[1].v_tst) result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    result = libxsmm_matdiff(di + 2, LIBXSMM_DATATYPE(ELEMTYPE), 3/*m*/, 1/*n*/,
      x/*ref*/, y/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);
  }
  if (EXIT_SUCCESS == result) {
    const double epsilon = libxsmm_matdiff_epsilon(di + 2);
    libxsmm_matdiff_reduce(&diff, di + 2);
    /* Epsilon (combined) */
    if (0.0000001 < LIBXSMM_ABS(epsilon - 0.0223103)) result = EXIT_FAILURE;
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(di[2].norm1_abs - 3.1000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[2].norm1_rel - 0.0281818)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(di[2].normi_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[2].normi_rel - 0.0200000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(di[2].normf_rel - 0.0222918)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(di[2].l2_abs - 2.2383029)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[2].l2_rel - 0.2438908)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(di[2].l1_ref - 110.00)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[2].l1_tst - 111.10)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(di[2].linf_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[2].linf_rel - 0.2222222)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(di[2].rsq - 0.9991717)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (2 != di[2].m || 0 != di[2].n) result = EXIT_FAILURE;
    if (x[3*di[2].n+di[2].m] != di[2].v_ref) result = EXIT_FAILURE;
    if (y[3*di[2].n+di[2].m] != di[2].v_tst) result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    result = libxsmm_matdiff(di + 3, LIBXSMM_DATATYPE(ELEMTYPE), 3/*m*/, 1/*n*/,
      r/*ref*/, t/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);
  }
  if (EXIT_SUCCESS == result) {
    const double epsilon = libxsmm_matdiff_epsilon(di + 3);
    libxsmm_matdiff_reduce(&diff, di + 3);
    /* Epsilon (combined) */
    if (0.0000001 < LIBXSMM_ABS(epsilon - 0.0006004)) result = EXIT_FAILURE;
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(di[3].norm1_abs - 0.0400000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[3].norm1_rel - 0.0400000)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(di[3].normi_abs - 0.0200000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[3].normi_rel - 0.0200000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(di[3].normf_rel - 0.0006000)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(di[3].l2_abs - 0.0244949)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[3].l2_rel - 0.0244949)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(di[3].l1_ref - 0.00)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[3].l1_tst - 0.04)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(di[3].linf_abs - 0.0200000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[3].linf_rel - 0.0200000)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(di[3].rsq - 0.9994000)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (1 != di[3].m || 0 != di[3].n) result = EXIT_FAILURE;
    if (r[3*di[3].n+di[3].m] != di[3].v_ref) result = EXIT_FAILURE;
    if (t[3*di[3].n+di[3].m] != di[3].v_tst) result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    result = libxsmm_matdiff(di + 4, LIBXSMM_DATATYPE(ELEMTYPE), 3/*m*/, 1/*n*/,
      t/*ref*/, r/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);
  }
  if (EXIT_SUCCESS == result) {
    const double epsilon = libxsmm_matdiff_epsilon(di + 4);
    /* intentionally not considered: libxsmm_matdiff_reduce(&diff, di + 4) */
    /* Epsilon (combined) */
    if (0.0000001 < LIBXSMM_ABS(epsilon - 0.0244949)) result = EXIT_FAILURE;
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(di[4].norm1_abs - 0.0400000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[4].norm1_rel - 1.0000000)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(di[4].normi_abs - 0.0200000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[4].normi_rel - 1.0000000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(di[4].normf_rel - 1.0000000)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(di[4].l2_abs - 0.0244949)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[4].l2_rel - 1.7320508)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(di[4].l1_ref - 0.04)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[4].l1_tst - 0.00)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(di[4].linf_abs - 0.0200000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[4].linf_rel - 1.0000000)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(di[4].rsq + 0.0000000)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (1 != di[4].m || 0 != di[4].n) result = EXIT_FAILURE;
    if (t[3*di[4].n+di[4].m] != di[4].v_ref) result = EXIT_FAILURE;
    if (r[3*di[4].n+di[4].m] != di[4].v_tst) result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    result = libxsmm_matdiff(di + 5, LIBXSMM_DATATYPE(ELEMTYPE), 3/*m*/, 1/*n*/,
      r/*ref*/, r/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);
  }
  if (EXIT_SUCCESS == result) {
    const double epsilon = libxsmm_matdiff_epsilon(di + 5);
    libxsmm_matdiff_reduce(&diff, di + 5);
    /* Epsilon (combined) */
    if (0.0000001 < LIBXSMM_ABS(epsilon - 0.0000000)) result = EXIT_FAILURE;
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(di[5].norm1_abs - 0.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[5].norm1_rel - 0.0000000)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(di[5].normi_abs - 0.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[5].normi_rel - 0.0000000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(di[5].normf_rel - 0.0000000)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(di[5].l2_abs - 0.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[5].l2_rel - 0.0000000)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(di[5].l1_ref - 0.00)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[5].l1_tst - 0.00)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(di[5].linf_abs - 0.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(di[5].linf_rel - 0.0000000)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(di[5].rsq - 1.0000000)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (-1 != di[5].m || -1 != di[5].n) result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    const double epsilon = libxsmm_matdiff_epsilon(&diff);
    /* Epsilon (combined) */
    if (0.0000001 < LIBXSMM_ABS(epsilon - 0.1132714)) result = EXIT_FAILURE;
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_abs - 3.1000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_rel - 0.0281818)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000002 < LIBXSMM_ABS(diff.normi_abs - 2.4400000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.normi_rel - 0.0976000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(diff.normf_rel - 0.1074954)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l2_abs - 2.2383029)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.l2_rel - 0.2438908)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l1_ref - 266.00)) result = EXIT_FAILURE;
    if (0.0000007 < LIBXSMM_ABS(diff.l1_tst - 267.90)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.linf_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.linf_rel - 0.2222222)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(diff.rsq - 0.9490077)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (2 != diff.m || 2 != diff.n) result = EXIT_FAILURE;
    if (a[3*diff.m+diff.n] != diff.v_ref) result = EXIT_FAILURE;
    if (b[3*diff.m+diff.n] != diff.v_tst) result = EXIT_FAILURE;
  }

  return result;
}
