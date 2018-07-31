/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
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
    const double ri = real_ref[i*ldr+j], ti = (0 != real_tst ? real_tst[i*ldt+j] : 0);
    const double di = (0 != real_tst ? (ri < ti ? (ti - ri) : (ri - ti)) : 0);
    const double ra = LIBXSMM_ABS(ri), ta = LIBXSMM_ABS(ti);
    if (LIBXSMM_NOTNAN(ta) && INFINITY > ta) {
      /* maximum absolute error and location */
      if (info->linf_abs < di) {
        info->linf_abs = di;
        info->linf_abs_m = j;
        info->linf_abs_n = i;
      }

      /* maximum error relative to current value */
      if (0 < ra) {
        const double dri = di / ra;
        if (info->linf_rel < dri) info->linf_rel = dri;
        /* sum of relative differences */
        v0 = dri * dri;
        if (INFINITY > v0) {
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
      if (INFINITY > v0) {
        v0 -= compf;
        v1 = info->l2_abs + v0;
        compf = (v1 - info->l2_abs) - v0;
        info->l2_abs = v1;
      }
    }
    else { /* NaN */
      result = EXIT_FAILURE;
      break;
    }
  }

  if (EXIT_SUCCESS == result) {
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

if (EXIT_SUCCESS == result) {
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
      const double ra = LIBXSMM_ABS(ri), ta = LIBXSMM_ABS(ti);

      /* column-wise sum of reference values with Kahan compensation */
      double v0 = ra - compri, v1 = normri + v0;
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
}

