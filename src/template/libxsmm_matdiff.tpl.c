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

const LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE *const real_ref = (const LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE*)ref;
const LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE *const real_tst = (const LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE*)tst;
double compf = 0, compfr = 0, compft = 0, normfr = 0, normft = 0, normr = 0, normt = 0;
double normrc = 0, normtc = 0, compr = 0, compt = 0, compd = 0;
libxsmm_blasint i, j;

for (i = 0; i < nn; ++i) {
  double comprj = 0, comptj = 0, compij = 0;
  double normrj = 0, normtj = 0, normij = 0;
  double v0, v1;

  for (j = 0; j < mm; ++j) {
    const double ti = (NULL != real_tst ? real_tst[i*ldt+j] : 0);
    const double ri = real_ref[i*ldr+j];
    const double ta = LIBXSMM_ABS(ti);
    const double ra = LIBXSMM_ABS(ri);

    /* minimum/maximum of reference set */
    if (ri < info->min_ref) info->min_ref = ri;
    if (ri > info->max_ref) info->max_ref = ri;

    if (LIBXSMM_NOTNAN(ti) && (inf > ta || ta == ra)) {
      const double di = (NULL != real_tst ? (ri < ti ? (ti - ri) : (ri - ti)) : 0);

      /* minimum/maximum of test set */
      if (ti < info->min_tst) info->min_tst = ti;
      if (ti > info->max_tst) info->max_tst = ti;

      /* maximum absolute error and location */
      if (info->linf_abs < di) {
        info->linf_abs = di;
        info->v_ref = ri;
        info->v_tst = ti;
        info->m = j;
        info->n = i;
      }

      /* maximum error relative to current value */
      if (0 < ra) {
        const double dri = di / ra;
        if (info->linf_rel < dri) info->linf_rel = dri;
        /* sum of relative differences */
        v0 = dri * dri;
        if (inf > v0) {
          v0 -= compd;
          v1 = info->l2_rel + v0;
          compd = (v1 - info->l2_rel) - v0;
          info->l2_rel = v1;
        }
      }

      /* row-wise sum of reference values with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(ra, &normrj, &comprj);

      /* row-wise sum of test values with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(ta, &normtj, &comptj);

      /* row-wise sum of differences with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(di, &normij, &compij);

      /* Froebenius-norm of reference matrix with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(ri * ri, &normfr, &compfr);

      /* Froebenius-norm of test matrix with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(ti * ti, &normft, &compft);

      /* Froebenius-norm of differences with Kahan compensation */
      v0 = di * di;
      if (inf > v0) {
        LIBXSMM_PRAGMA_FORCEINLINE
        libxsmm_kahan_sum(v0, &info->l2_abs, &compf);
      }
    }
    else { /* NaN */
      result_nan = ((LIBXSMM_NOTNAN(ri) && inf > ra) ? 1 : 2);
      info->m = j; info->n = i;
      info->v_ref = ri;
      info->v_tst = ti;
      break;
    }
  }

  if (0 == result_nan) {
    /* summarize reference values */
    LIBXSMM_PRAGMA_FORCEINLINE
    libxsmm_kahan_sum(normrj, &info->l1_ref, &compr);

    /* summarize test values */
    LIBXSMM_PRAGMA_FORCEINLINE
    libxsmm_kahan_sum(normtj, &info->l1_tst, &compt);

    /* calculate Infinity-norm of differences */
    if (info->normi_abs < normij) info->normi_abs = normij;
    /* calculate Infinity-norm of reference/test values */
    if (normr < normrj) normr = normrj;
    if (normt < normtj) normt = normtj;
  }
  else {
    break;
  }
}

if (0 == result_nan) {
  double compr_var = 0, compt_var = 0;

  /* initial variance */
  assert(0 == info->var_ref); /* !LIBXSMM_ASSERT */
  assert(0 == info->var_tst); /* !LIBXSMM_ASSERT */

  if (0 != ntotal) { /* final average */
    info->avg_ref = info->l1_ref / ntotal;
    info->avg_tst = info->l1_tst / ntotal;
  }

  /* Infinity-norm relative to reference */
  info->normi_rel = LIBXSMM_MATDIFF_DIV(info->normi_abs, normr, normt);
  /* Froebenius-norm relative to reference */
  info->normf_rel = LIBXSMM_MATDIFF_DIV(info->l2_abs, normfr, normft);

  for (j = 0; j < mm; ++j) {
    double compri = 0, compti = 0, comp1 = 0;
    double normri = 0, normti = 0, norm1 = 0;

    for (i = 0; i < nn; ++i) {
      const double ri = real_ref[i*ldr + j], ti = (NULL != real_tst ? real_tst[i*ldt + j] : 0);
      const double di = (NULL != real_tst ? (ri < ti ? (ti - ri) : (ri - ti)) : 0);
      const double rd = ri - info->avg_ref, td = ti - info->avg_tst;
      const double ra = LIBXSMM_ABS(ri), ta = LIBXSMM_ABS(ti);

      /* variance of reference set with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(rd * rd, &info->var_ref, &compr_var);

      /* variance of test set with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(td * td, &info->var_tst, &compt_var);

      /* column-wise sum of reference values with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(ra, &normri, &compri);

      /* column-wise sum of test values with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(ta, &normti, &compti);

      /* column-wise sum of differences with Kahan compensation */
      LIBXSMM_PRAGMA_FORCEINLINE
      libxsmm_kahan_sum(di, &norm1, &comp1);
    }

    /* calculate One-norm of differences */
    if (info->norm1_abs < norm1) info->norm1_abs = norm1;
    /* calculate One-norm of reference/test values */
    if (normrc < normri) normrc = normri;
    if (normtc < normti) normtc = normti;
  }

  /* One-norm relative to reference */
  info->norm1_rel = LIBXSMM_MATDIFF_DIV(info->norm1_abs, normrc, normtc);
}
