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

#if !defined(ITYPE)
# define ITYPE double
#endif


int main(void)
{
  int result = EXIT_SUCCESS;
  libxsmm_matdiff_info da, db, dc, diff;
  /* http://www.netlib.org/lapack/lug/node75.html */
  const ITYPE a[] = {
    (ITYPE)1.00, (ITYPE)2.00, (ITYPE)3.00,
    (ITYPE)4.00, (ITYPE)5.00, (ITYPE)6.00,
    (ITYPE)7.00, (ITYPE)8.00, (ITYPE)10.0
  };
  const ITYPE b[] = {
    (ITYPE)0.44, (ITYPE)2.36, (ITYPE)3.04,
    (ITYPE)3.09, (ITYPE)5.87, (ITYPE)6.66,
    (ITYPE)7.36, (ITYPE)7.77, (ITYPE)9.07
  };
  const ITYPE x[] = {
    (ITYPE)1.00, (ITYPE)100.0, (ITYPE)9.00
  };
  const ITYPE y[] = {
    (ITYPE)1.10, (ITYPE)99.00, (ITYPE)11.0
  };

  /* no need to clear da, db, and dc; just the accumulator (diff) */
  libxsmm_matdiff_clear(&diff);

  result = libxsmm_matdiff(&da, LIBXSMM_DATATYPE(ITYPE), 3/*m*/, 3/*n*/,
    a/*ref*/, b/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);

  if (EXIT_SUCCESS == result) {
    libxsmm_matdiff_reduce(&diff, &da);
    /* One-norm */
    if (0.0000003 < LIBXSMM_ABS(da.norm1_abs - 1.8300000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(da.norm1_rel - 0.0963158)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000002 < LIBXSMM_ABS(da.normi_abs - 2.4400000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(da.normi_rel - 0.0976000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(da.normf_rel - 0.1074954)) result = EXIT_FAILURE;
    /* L2-norm */
    if (0.0000002 < LIBXSMM_ABS(da.l2_abs - 1.8742465)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(da.l2_rel - 0.6726295)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(da.l1_ref - 46.00)) result = EXIT_FAILURE;
    if (0.0000007 < LIBXSMM_ABS(da.l1_tst - 45.66)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000004 < LIBXSMM_ABS(da.linf_abs - 0.9300000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(da.linf_rel - 0.5600000)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(da.rsq - 0.9490077)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (2 != da.m || 2 != da.n) result = EXIT_FAILURE;
    if (a[3*da.m+da.n] != da.v_ref) result = EXIT_FAILURE;
    if (b[3*da.m+da.n] != da.v_tst) result = EXIT_FAILURE;
  }

  result = libxsmm_matdiff(&db, LIBXSMM_DATATYPE(ITYPE), 1/*m*/, 3/*n*/,
    x/*ref*/, y/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);

  if (EXIT_SUCCESS == result) {
    libxsmm_matdiff_reduce(&diff, &db);
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(db.norm1_abs - 3.1000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(db.norm1_rel - 0.0281818)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(db.normi_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(db.normi_rel - 0.0200000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(db.normf_rel - 0.0222918)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(db.l2_abs - 2.2383029)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(db.l2_rel - 0.2438908)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(db.l1_ref - 110.00)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(db.l1_tst - 111.10)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(db.linf_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(db.linf_rel - 0.2222222)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(db.rsq - 0.9991717)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (0 != db.m || 2 != db.n) result = EXIT_FAILURE;
    if (x[3*db.m+db.n] != db.v_ref) result = EXIT_FAILURE;
    if (y[3*db.m+db.n] != db.v_tst) result = EXIT_FAILURE;
  }

  result = libxsmm_matdiff(&dc, LIBXSMM_DATATYPE(ITYPE), 3/*m*/, 1/*n*/,
    x/*ref*/, y/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);

  if (EXIT_SUCCESS == result) {
    libxsmm_matdiff_reduce(&diff, &dc);
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(dc.norm1_abs - 3.1000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(dc.norm1_rel - 0.0281818)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(dc.normi_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(dc.normi_rel - 0.0200000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(dc.normf_rel - 0.0222918)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(dc.l2_abs - 2.2383029)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(dc.l2_rel - 0.2438908)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(dc.l1_ref - 110.00)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(dc.l1_tst - 111.10)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(dc.linf_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(dc.linf_rel - 0.2222222)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(dc.rsq - 0.9991717)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (2 != dc.m || 0 != dc.n) result = EXIT_FAILURE;
    if (x[3*dc.n+dc.m] != dc.v_ref) result = EXIT_FAILURE;
    if (y[3*dc.n+dc.m] != dc.v_tst) result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_abs - 3.1000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_rel - 0.0281818)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.normi_abs - 2.4400000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.normi_rel - 0.0976000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(diff.normf_rel - 0.1074954)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l2_abs - 2.2383029)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.l2_rel - 0.2438908)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l1_ref - 266.00)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.l1_tst - 267.86)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.linf_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.linf_rel - 0.2222222)) result = EXIT_FAILURE;
    /* R-squared */
    if (0.0000001 < LIBXSMM_ABS(diff.rsq - 0.9991717)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (0 != diff.m || 2 != diff.n) result = EXIT_FAILURE;
    if (x[3*diff.m+diff.n] != diff.v_ref) result = EXIT_FAILURE;
    if (y[3*diff.m+diff.n] != diff.v_tst) result = EXIT_FAILURE;
  }

  return result;
}

