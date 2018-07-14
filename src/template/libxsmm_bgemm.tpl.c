/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
LIBXSMM_VLA_DECL(2, LIBXSMM_BGEMM_TEMPLATE_TYPE_C, l_out, (LIBXSMM_BGEMM_TEMPLATE_TYPE_C*)(((char*)handle->buffer) +
  ltid * LIBXSMM_UP2(handle->bm * handle->bn * sizeof(LIBXSMM_BGEMM_TEMPLATE_TYPE_C), LIBXSMM_CACHELINE)), handle->bm);
LIBXSMM_VLA_DECL(4, const LIBXSMM_BGEMM_TEMPLATE_TYPE_AB, real_a, (const LIBXSMM_BGEMM_TEMPLATE_TYPE_AB*)a, handle->kb, handle->bk, handle->bm);
LIBXSMM_VLA_DECL(4, const LIBXSMM_BGEMM_TEMPLATE_TYPE_AB, real_b, (const LIBXSMM_BGEMM_TEMPLATE_TYPE_AB*)b, handle->kb, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, LIBXSMM_BGEMM_TEMPLATE_TYPE_C, real_c, (LIBXSMM_BGEMM_TEMPLATE_TYPE_C*)c, handle->mb, handle->bn, handle->bm);

const LIBXSMM_MMFUNCTION_TYPE2(LIBXSMM_BGEMM_TEMPLATE_TYPE_AB, LIBXSMM_BGEMM_TEMPLATE_TYPE_C) kernel =
        handle->kernel.LIBXSMM_TPREFIX2(LIBXSMM_BGEMM_TEMPLATE_TYPE_AB, LIBXSMM_BGEMM_TEMPLATE_TYPE_C, mm);
const LIBXSMM_MMFUNCTION_TYPE2(LIBXSMM_BGEMM_TEMPLATE_TYPE_AB, LIBXSMM_BGEMM_TEMPLATE_TYPE_C) kernel_pf =
        handle->kernel_pf.LIBXSMM_TPREFIX2(LIBXSMM_BGEMM_TEMPLATE_TYPE_AB, LIBXSMM_BGEMM_TEMPLATE_TYPE_C, mm);

const libxsmm_blasint b_m1 = handle->b_m1;
const libxsmm_blasint b_n1 = handle->b_n1;
const libxsmm_blasint b_k1 = handle->b_k1;
const libxsmm_blasint b_k2 = handle->b_k2;

const libxsmm_blasint mm = handle->m / b_m1;
const libxsmm_blasint nn = handle->n / b_n1;
const libxsmm_blasint kk = handle->k / b_k1;

const libxsmm_blasint nw_i = mm / handle->bm;
const libxsmm_blasint nw_j = nn / handle->bn;
const libxsmm_blasint nw_k = kk / handle->bk;
const libxsmm_blasint nw = nw_i * nw_j;

libxsmm_blasint m, n, k, mb, nb, kb;
libxsmm_blasint ki, kj, w_i, ki2;
libxsmm_blasint nw_k2 = nw_k;

/* TODO: take transa and transb into account (flags) */

for (ki = 0; ki < handle->bn; ++ki) {
  LIBXSMM_PRAGMA_SIMD
  for (kj = 0; kj < handle->bm; ++kj) {
    LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm) = 0;
  }
}

for (mb = 0, m = 0; mb < b_m1; ++mb, m += nw_i) {
  for (nb = 0, n = 0; nb < b_n1; ++nb, n += nw_j) {
    for (kb = 0, k = 0; kb < b_k1; ++kb, k += nw_k2) {
      const libxsmm_blasint nw_k3 = nw_k / b_k2;
      const libxsmm_blasint nw2 = nw * nw_k3;
      const libxsmm_blasint s = (ltid * nw2) / handle->nthreads;
      const libxsmm_blasint e = ((ltid + 1) * nw2) / handle->nthreads;
      libxsmm_blasint o_i2 = 0, o_j2 = 0;
      nw_k2 = nw_k3;

      for (w_i = s; w_i < e; ++w_i) {
        libxsmm_blasint i2 = 0, j2 = 0, k2 = 0;
        internal_bgemm_order(handle->order, w_i, nw_i, nw_j, nw_k2, &i2, &j2, &k2);
        i2 += m; j2 += n; k2 += k;

        if (w_i == s) {
          o_i2 = i2;
          o_j2 = j2;
        }
        else {
          if (o_i2 != i2 || o_j2 != j2) {
            libxsmm_bgemm_lock *const lock = &LIBXSMM_VLA_ACCESS(2, locks, o_i2, o_j2, handle->nb);
            LIBXSMM_ATOMIC_ACQUIRE(&lock->state, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_RELAXED);
            for (ki = 0; ki < handle->bn; ++ki) {
              LIBXSMM_PRAGMA_SIMD
              for (kj = 0; kj < handle->bm; ++kj) {
                LIBXSMM_VLA_ACCESS(4, real_c, o_j2, o_i2, ki, kj, handle->mb, handle->bn, handle->bm) +=
                LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm);
              }
            }
            LIBXSMM_ATOMIC_RELEASE(&lock->state, LIBXSMM_ATOMIC_RELAXED);
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

        if (0 != kernel_pf) { /* prefetch */
          for (ki2 = 0, ki = (b_k2 * k2); ki2 < b_k2; ++ki2, ++ki) {
            if (k2 < (nw_k - 2)) { /* prefetch */
              kernel_pf(&LIBXSMM_VLA_ACCESS(4, real_a, i2, ki, 0, 0, handle->kb, handle->bk, handle->bm),
                        &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk),
                        &LIBXSMM_VLA_ACCESS(2, l_out, 0, 0, handle->bm),
                        &LIBXSMM_VLA_ACCESS(4, real_a, i2, ki+1, 0, 0, handle->kb, handle->bk, handle->bm),
                        &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki+1, 0, 0, handle->kb, handle->bn, handle->bk), NULL);
            }
            else { /* avoid prefetching OOB */
              kernel(&LIBXSMM_VLA_ACCESS(4, real_a, i2, ki, 0, 0, handle->kb, handle->bk, handle->bm),
                     &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk),
                     &LIBXSMM_VLA_ACCESS(2, l_out, 0, 0, handle->bm));
            }
          }
        }
        else { /* no prefetch */
          for (ki2 = 0, ki = (b_k2 * k2); ki2 < b_k2; ++ki2, ++ki) {
            kernel(&LIBXSMM_VLA_ACCESS(4, real_a, i2, ki, 0, 0, handle->kb, handle->bk, handle->bm),
                   &LIBXSMM_VLA_ACCESS(4, real_b, j2, ki, 0, 0, handle->kb, handle->bn, handle->bk),
                   &LIBXSMM_VLA_ACCESS(2, l_out, 0, 0, handle->bm));
          }
        }

        if (w_i == (e - 1)) {
          libxsmm_bgemm_lock* lock;
          o_i2 = i2;
          o_j2 = j2;

          lock = &LIBXSMM_VLA_ACCESS(2, locks, o_i2, o_j2, handle->nb);
          LIBXSMM_ATOMIC_ACQUIRE(&lock->state, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_RELAXED);
          for (ki = 0; ki < handle->bn; ++ki) {
            LIBXSMM_PRAGMA_SIMD
            for (kj = 0; kj < handle->bm; ++kj) {
              LIBXSMM_VLA_ACCESS(4, real_c, o_j2, o_i2, ki, kj, handle->mb, handle->bn, handle->bm) +=
              LIBXSMM_VLA_ACCESS(2, l_out, ki, kj, handle->bm);
            }
          }
          LIBXSMM_ATOMIC_RELEASE(&lock->state, LIBXSMM_ATOMIC_RELAXED);
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

