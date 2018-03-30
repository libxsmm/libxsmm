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
#include <libxsmm_source.h>
#include <float.h>

#if !defined(ITYPE)
# define ITYPE double
#endif


int main(void)
{
  int result = EXIT_SUCCESS;
  libxsmm_matdiff_info diff;
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

  result = libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), 3/*m*/, 3/*n*/,
    a/*ref*/, b/*tst*/, NULL/*ldref*/, NULL/*ldtst*/, &diff);

  if (EXIT_SUCCESS == result) {
    /* One-norm */
    if (0.0000003 < LIBXSMM_ABS(diff.norm1_abs - 1.8300000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_rel - 0.0963158)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000002 < LIBXSMM_ABS(diff.normi_abs - 2.4400000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.normi_rel - 0.0976000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(diff.normf_rel - 0.1074954)) result = EXIT_FAILURE;
    /* L2-norm */
    if (0.0000002 < LIBXSMM_ABS(diff.l2_abs - 1.8742465)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.l2_rel - 0.6726295)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l1_ref - 46.00)) result = EXIT_FAILURE;
    if (0.0000007 < LIBXSMM_ABS(diff.l1_tst - 45.66)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000004 < LIBXSMM_ABS(diff.linf_abs - 0.9300000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.linf_rel - 0.5600000)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (2 != diff.linf_abs_m) result = EXIT_FAILURE;
    if (2 != diff.linf_abs_n) result = EXIT_FAILURE;
  }

  result = libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), 1/*m*/, 3/*n*/,
    x/*ref*/, y/*tst*/, NULL/*ldref*/, NULL/*ldtst*/, &diff);

  if (EXIT_SUCCESS == result) {
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_abs - 3.1000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_rel - 0.0281818)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.normi_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.normi_rel - 0.0200000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(diff.normf_rel - 0.0222918)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l2_abs - 2.2383029)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.l2_rel - 0.2438908)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l1_ref - 110.0)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.l1_tst - 111.1)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.linf_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.linf_rel - 0.2222222)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (0 != diff.linf_abs_m) result = EXIT_FAILURE;
    if (2 != diff.linf_abs_n) result = EXIT_FAILURE;
  }

  result = libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), 3/*m*/, 1/*n*/,
    x/*ref*/, y/*tst*/, NULL/*ldref*/, NULL/*ldtst*/, &diff);

  if (EXIT_SUCCESS == result) {
    /* One-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_abs - 3.1000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.norm1_rel - 0.0281818)) result = EXIT_FAILURE;
    /* Infinity-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.normi_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.normi_rel - 0.0200000)) result = EXIT_FAILURE;
    /* Froebenius-norm (relative) */
    if (0.0000001 < LIBXSMM_ABS(diff.normf_rel - 0.0222918)) result = EXIT_FAILURE;
    /** L2-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l2_abs - 2.2383029)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.l2_rel - 0.2438908)) result = EXIT_FAILURE;
    /** L1-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.l1_ref - 110.0)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.l1_tst - 111.1)) result = EXIT_FAILURE;
    /* Linf-norm */
    if (0.0000001 < LIBXSMM_ABS(diff.linf_abs - 2.0000000)) result = EXIT_FAILURE;
    if (0.0000001 < LIBXSMM_ABS(diff.linf_rel - 0.2222222)) result = EXIT_FAILURE;
    /* Location of maximum absolute error */
    if (2 != diff.linf_abs_m) result = EXIT_FAILURE;
    if (0 != diff.linf_abs_n) result = EXIT_FAILURE;
  }

  return result;
}

