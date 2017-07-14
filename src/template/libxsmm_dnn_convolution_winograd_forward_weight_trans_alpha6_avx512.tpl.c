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
/* Kunal Banerjee (Intel Corp.), Jongsoo Park (Intel Corp.)
******************************************************************************/

LIBXSMM_VLA_DECL(6, float, input, wp, handle->blocksifm, 3, 3, TDVLEN, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, twp, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN);
float Fw[ALPHA][ALPHA][TDVLEN][TDVLEN];
float F[3][3][TDVLEN][TDVLEN];
unsigned int i, j;
int ifm2;
const __m512 rcp4  = _mm512_set1_ps(1.0f/4.0f);
const __m512 rcp6  = _mm512_set1_ps(1.0f/6.0f);
const __m512 rcp12 = _mm512_set1_ps(1.0f/12.0f);
const __m512 rcp24 = _mm512_set1_ps(1.0f/24.0f);
__m512 T[ALPHA][3];

for (ifm2 = 0; ifm2 < TDVLEN; ifm2++)
{
  /*LIBXSMM_PRAGMA_UNROLL_N(3)*/
  for (i = 0; i < 3; i++)
  {
    __m512 f0, f1, f2;
    f0 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(6, input, 0, 0, 0, i, ifm2, 0, handle->blocksifm, 3, 3, TDVLEN, TDVLEN));
    f1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(6, input, 0, 0, 1, i, ifm2, 0, handle->blocksifm, 3, 3, TDVLEN, TDVLEN));
    f2 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(6, input, 0, 0, 2, i, ifm2, 0, handle->blocksifm, 3, 3, TDVLEN, TDVLEN));

    __m512 t0, t1, t2;
    t0 = _mm512_mul_ps(rcp6, f2);
    t1 = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_fmadd_ps(rcp6, f0, t0));
    t2 = _mm512_fmadd_ps(rcp24, f0, t0);

    T[0][i] = _mm512_mul_ps(rcp4, f0);
    T[1][i] = _mm512_fnmadd_ps(rcp6, f1, t1);
    T[2][i] = _mm512_fmadd_ps(rcp6, f1, t1);
    T[3][i] = _mm512_fmadd_ps(rcp12, f1, t2);
    T[4][i] = _mm512_fnmadd_ps(rcp12, f1, t2);
    T[5][i] = f2;
  }

  /*LIBXSMM_PRAGMA_UNROLL_N(ALPHA)*/
  for (j = 0; j < ALPHA; j++)
  {
    __m512 t0, t1, t2;
    t0 = _mm512_mul_ps(rcp6, T[j][2]);
    t1 = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_fmadd_ps(rcp6, T[j][0], t0));
    t2 = _mm512_fmadd_ps(rcp24, T[j][0], t0);

    /* Since we are using streaming store to save read BW and don't need HW prefetcher,
     * the loop order doesn't need to make these writes accesses contiguous
     */
    LIBXSMM_INTRINSICS_MM512_STREAM_PS(
        &LIBXSMM_VLA_ACCESS(5, output, j, 0, 0, ifm2, 0, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN),
        _mm512_mul_ps(rcp4, T[j][0]));
    LIBXSMM_INTRINSICS_MM512_STREAM_PS(
        &LIBXSMM_VLA_ACCESS(5, output, j, 1, 0, ifm2, 0, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN),
        _mm512_fnmadd_ps(rcp6, T[j][1], t1));
    LIBXSMM_INTRINSICS_MM512_STREAM_PS(
        &LIBXSMM_VLA_ACCESS(5, output, j, 2, 0, ifm2, 0, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN),
        _mm512_fmadd_ps(rcp6, T[j][1], t1));
    LIBXSMM_INTRINSICS_MM512_STREAM_PS(
        &LIBXSMM_VLA_ACCESS(5, output, j, 3, 0, ifm2, 0, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN),
        _mm512_fmadd_ps(rcp12, T[j][1], t2));
    LIBXSMM_INTRINSICS_MM512_STREAM_PS(
        &LIBXSMM_VLA_ACCESS(5, output, j, 4, 0, ifm2, 0, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN),
        _mm512_fnmadd_ps(rcp12, T[j][1], t2));
    LIBXSMM_INTRINSICS_MM512_STREAM_PS(
        &LIBXSMM_VLA_ACCESS(5, output, j, 5, 0, ifm2, 0, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN),
        T[j][2]);
  }
}
