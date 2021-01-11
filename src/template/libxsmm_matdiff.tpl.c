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
    const double ti = (0 != real_tst ? real_tst[i*ldt+j] : 0);
    const double ri = real_ref[i*ldr+j];
    const double ta = LIBXSMM_ABS(ti);
    const double ra = LIBXSMM_ABS(ri);

    /* minimum/maximum of reference set */
    if (ri < info->min_ref) info->min_ref = ri;
    if (ri > info->max_ref) info->max_ref = ri;

    if (LIBXSMM_NOTNAN(ti) && inf > ta) {
      const double di = (0 != real_tst ? (ri < ti ? (ti - ri) : (ri - ti)) : 0);

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
      v0 = ra - comprj; v1 = normrj + v0;
      comprj = (v1 - normrj) - v0;
      normrj = v1;

      /* row-wise sum of test values with Kahan compensation */
      v0 = ta - comptj; v1 = normtj + v0;
      comptj = (v1 - normtj) - v0;
      normtj = v1;

      /* row-wise sum of differences with Kahan compensation */
      v0 = di - compij; v1 = normij + v0;
      compij = (v1 - normij) - v0;
      normij = v1;

      /* Froebenius-norm of reference matrix with Kahan compensation */
      v0 = ri * ri - compfr; v1 = normfr + v0;
      compfr = (v1 - normfr) - v0;
      normfr = v1;

      /* Froebenius-norm of test matrix with Kahan compensation */
      v0 = ti * ti - compft; v1 = normft + v0;
      compft = (v1 - normft) - v0;
      normft = v1;

      /* Froebenius-norm of differences with Kahan compensation */
      v0 = di * di;
      if (inf > v0) {
        v0 -= compf;
        v1 = info->l2_abs + v0;
        compf = (v1 - info->l2_abs) - v0;
        info->l2_abs = v1;
      }
    }
    else { /* NaN */
      info->m = j;
      info->n = i;
      result_nan = ((LIBXSMM_NOTNAN(ri) && inf > ra) ? 1 : 2);
      break;
    }
  }

  if (0 == result_nan) {
    /* summarize reference values */
    v0 = normrj - compr; v1 = info->l1_ref + v0;
    compr = (v1 - info->l1_ref) - v0;
    info->l1_ref = v1;

    /* summarize test values */
    v0 = normtj - compt; v1 = info->l1_tst + v0;
    compt = (v1 - info->l1_tst) - v0;
    info->l1_tst = v1;

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
  const libxsmm_blasint size = mm * nn;
  double compr_var = 0, compt_var = 0;

  /* initial variance */
  LIBXSMM_ASSERT(0 == info->var_ref);
  LIBXSMM_ASSERT(0 == info->var_tst);

  if (0 != size) { /* final average */
    info->avg_ref = info->l1_ref / size;
    info->avg_tst = info->l1_tst / size;
  }
  /* Infinity-norm relative to reference */
  if (0 < normr) {
    info->normi_rel = info->normi_abs / normr;
  }
  else if (0 < normt) { /* relative to test */
    info->normi_rel = info->normi_abs / normt;
  }
  else { /* should not happen */
    info->normi_rel = 0;
  }

  /* Froebenius-norm relative to reference */
  if (0 < normfr) {
    info->normf_rel = info->l2_abs / normfr;
  }
  else if (0 < normft) { /* relative to test */
    info->normf_rel = info->l2_abs / normft;
  }
  else { /* should not happen */
    info->normf_rel = 0;
  }

  for (j = 0; j < mm; ++j) {
    double compri = 0, compti = 0, comp1 = 0;
    double normri = 0, normti = 0, norm1 = 0;

    for (i = 0; i < nn; ++i) {
      const double ri = real_ref[i*ldr + j], ti = (0 != real_tst ? real_tst[i*ldt + j] : 0);
      const double di = (0 != real_tst ? (ri < ti ? (ti - ri) : (ri - ti)) : 0);
      const double rd = ri - info->avg_ref, td = ti - info->avg_tst;
      const double ra = LIBXSMM_ABS(ri), ta = LIBXSMM_ABS(ti);

      /* variance of reference set with Kahan compensation */
      double v0 = rd * rd - compr_var, v1 = info->var_ref + v0;
      compr_var = (v1 - info->var_ref) - v0;
      info->var_ref = v1;

      /* variance of test set with Kahan compensation */
      v0 = td * td - compt_var; v1 = info->var_tst + v0;
      compt_var = (v1 - info->var_tst) - v0;
      info->var_tst = v1;

      /* column-wise sum of reference values with Kahan compensation */
      v0 = ra - compri; v1 = normri + v0;
      compri = (v1 - normri) - v0;
      normri = v1;

      /* column-wise sum of test values with Kahan compensation */
      v0 = ta - compti; v1 = normti + v0;
      compti = (v1 - normti) - v0;
      normti = v1;

      /* column-wise sum of differences with Kahan compensation */
      v0 = di - comp1; v1 = norm1 + v0;
      comp1 = (v1 - norm1) - v0;
      norm1 = v1;
    }

    /* calculate One-norm of differences */
    if (info->norm1_abs < norm1) info->norm1_abs = norm1;
    /* calculate One-norm of reference/test values */
    if (normrc < normri) normrc = normri;
    if (normtc < normti) normtc = normti;
  }

  /* One-norm relative to reference */
  if (0 < normrc) {
    info->norm1_rel = info->norm1_abs / normrc;
  }
  else if (0 < normtc) { /* relative to test */
    info->norm1_rel = info->norm1_abs / normtc;
  }
  else { /* should not happen */
    info->norm1_rel = 0;
  }
  if (0 != size) { /* final variance */
    info->var_ref /= size;
    info->var_tst /= size;
  }
}

