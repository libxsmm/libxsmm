/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/

LIBXSMM_VLA_DECL(2, libxsmm_bgemm_lock, locks, handle->locks, handle->nb);
/* TODO: pad thread-local buffer members by the size of a cache-line in order to avoid "Ping-Pong" */
LIBXSMM_VLA_DECL(2, LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C, l_out, (LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C*)(((char*)handle->buffer) + tid * handle->bm * handle->bn * handle->typesize), handle->bm);
LIBXSMM_VLA_DECL(4, const LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB, real_a, (const LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB*)a, handle->mb, handle->bk, handle->bm);
LIBXSMM_VLA_DECL(4, const LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB, real_b, (const LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB*)b, handle->kb, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C, real_c, (LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_C*)c, handle->mb, handle->bn, handle->bm);

const LIBXSMM_MMFUNCTION_TYPE(LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB) kernel = handle->kernel.LIBXSMM_TPREFIX(LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB, mm);
#if defined(LIBXSMM_BGEMM_PREFETCH)
const LIBXSMM_MMFUNCTION_TYPE(LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB) kernel_pf = handle->kernel_pf.LIBXSMM_TPREFIX(LIBXSMM_BGEMM_TEMPLATE_REAL_TYPE_AB, mm);
#endif

const libxsmm_blasint b_m1 = handle->b_m1;
const libxsmm_blasint b_n1 = handle->b_n1;
const libxsmm_blasint b_k1 = handle->b_k1;
const libxsmm_blasint b_k2 = handle->b_k2;

const libxsmm_blasint mm = handle->m / b_m1;
const libxsmm_blasint nn = handle->n / b_n1;
const libxsmm_blasint kk = handle->k / b_k1;

const libxsmm_blasint nw_i = mm / handle->bm;
const libxsmm_blasint nw_j = nn / handle->bn;
libxsmm_blasint nw_k = kk / handle->bk; /* TODO: check */
const libxsmm_blasint nw = nw_i * nw_j * nw_k;

libxsmm_blasint m, n, k, mb, nb, kb;
libxsmm_blasint ki, kj, w_i, _ki;

/* TODO: take transa and transb into account (flags) */

for (ki = 0; ki < handle->bn; ++ki) {
  LIBXSMM_PRAGMA_SIMD
  for (kj = 0; kj < handle->bm; ++kj) {
    LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm) = 0;
  }
}

for (mb = 0, m = 0; mb < b_m1; ++mb, m += nw_i) {
  for (nb = 0, n = 0; nb < b_n1; ++nb, n += nw_j) {
    for (kb = 0, k = 0; kb < b_k1; ++kb, k += nw_k) {
      int s = (tid * nw) / nthreads;
      int e = ((tid + 1) * nw) / nthreads;
      int o_i2 = 0, o_j2 = 0;
      nw_k /= b_k2; /* TODO: check */

      for (w_i = s; w_i < e; ++w_i) {
        int i2 = 0, j2 = 0, k2 = 0;
        internal_bgemm_order(handle->order, w_i, nw_i, nw_j, nw_k, &i2, &j2, &k2);

        i2 = m + i2;
        j2 = n + j2;
        k2 = k + k2;

        if (w_i == s) {
          o_i2 = i2;
          o_j2 = j2;
        }
        else {
          if ((o_i2 != i2) || (o_j2 != j2)) {
            libxsmm_bgemm_lock *const lock = &LIBXSMM_VLA_ACCESS(2, locks, o_i2, o_j2, handle->nb);
            LIBXSMM_SYNC_SET(lock->instance);
            for (ki = 0; ki < handle->bn; ++ki) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < handle->bm; ++kj) {
                LIBXSMM_VLA_ACCESS(4, real_c, o_j2, o_i2, ki, kj, handle->mb, handle->bn, handle->bm) +=
                LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm);
              }
            }
            LIBXSMM_SYNC_UNSET(lock->instance);
            for (ki = 0; ki < handle->bn; ++ki) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < handle->bm; ++kj) {
                LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm) = 0;
              }
            }
            o_i2 = i2;
            o_j2 = j2;
          }
        }
        for (_ki = 0, ki = (b_k2 * k2); _ki < b_k2 ; ++_ki, ++ki) {
#if !defined(LIBXSMM_BGEMM_PREFETCH)
          kernel(&LIBXSMM_VLA_ACCESS(4, real_a, ki, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                 &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk), l_out);
#else
          if (k2 < (nw_k - 2)) { /* avoid prefetching of untouched data */
            if (LIBXSMM_X86_AVX < libxsmm_target_archid) { /* TODO: check condition; though "__AVX2__" is included in AVX-512 */
              kernel_pf(&LIBXSMM_VLA_ACCESS(4, real_a, ki, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                        &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk), l_out,
                        &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki+1, 0, 0, handle->kb, handle->bn, handle->bk),
                        &LIBXSMM_VLA_ACCESS(4, real_a, ki+1, i2, 0, 0, handle->mb, handle->bk, handle->bm), NULL);              
            }
            else {
              kernel_pf(&LIBXSMM_VLA_ACCESS(4, real_a, ki, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                        &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk), l_out,
                        &LIBXSMM_VLA_ACCESS(4, real_a, ki+1, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                        &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki+1, 0, 0, handle->kb, handle->bn, handle->bk), NULL);              
            }
          }
          else {
            kernel(&LIBXSMM_VLA_ACCESS(4, real_a, ki, i2, 0, 0, handle->mb, handle->bk, handle->bm),
                   &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk), l_out);
          }
#endif
        }

        if (w_i == (e - 1)) {
          libxsmm_bgemm_lock* lock;
          o_i2 = i2;
          o_j2 = j2;

          lock = &LIBXSMM_VLA_ACCESS(2, locks, o_i2, o_j2, handle->nb);
          LIBXSMM_SYNC_SET(lock->instance);
          for (ki = 0; ki < handle->bn; ++ki) {
            LIBXSMM_PRAGMA_SIMD
            for (kj = 0; kj < handle->bm; ++kj) {
              LIBXSMM_VLA_ACCESS(4, real_c, o_j2, o_i2, ki, kj, handle->mb, handle->bn, handle->bm) +=
              LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm);
            }
          }
          LIBXSMM_SYNC_UNSET(lock->instance);
          for (ki = 0; ki < handle->bn; ++ki) {
            LIBXSMM_PRAGMA_SIMD
            for (kj = 0; kj < handle->bm; ++kj) {
              LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm) = 0;
            }
          }
        }
      }
    }
  }
}

