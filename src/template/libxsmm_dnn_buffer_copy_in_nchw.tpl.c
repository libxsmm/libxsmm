/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
/* Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/

/* use for-loops to potentially leverage NUMA in the future */
int i1, i2, i3, i4, i5, i6, i7;
int N = buffer->N;
int splits = buffer->splits;
int fmb = buffer->fmb;
int bfm = buffer->bfm;
int H = buffer->H;
int W = buffer->W;
int lpb = buffer->lpb;

LIBXSMM_VLA_DECL(7, element_type, handle_data, (element_type*)buffer->data, splits, fmb, H, W, bfm, lpb);
LIBXSMM_VLA_DECL(5, const element_type, user_data, (const element_type*)data, splits, fmb * bfm * lpb, H, W);

for (i1 = 0; i1 < N; ++i1) {
  for (i2 = 0; i2 < splits; ++i2) {
    for (i3 = 0; i3 < fmb; ++i3) {
      for (i4 = 0; i4 < H; ++i4) {
        for (i5 = 0; i5 < W; ++i5) {
          for (i6 = 0; i6 < bfm; ++i6) {
            for (i7 = 0; i7 < lpb; ++i7) {
              LIBXSMM_VLA_ACCESS(7, handle_data, i1, i2, i3, i4, i5, i6, i7, splits, fmb, H, W, bfm, lpb) =
              LIBXSMM_VLA_ACCESS(5, user_data, i1, i2, (i3*bfm*lpb) + (i6*lpb) + i7, i4, i5, splits, fmb * bfm * lpb, H, W);
            }
          }
        }
      }
    }
  }
}
