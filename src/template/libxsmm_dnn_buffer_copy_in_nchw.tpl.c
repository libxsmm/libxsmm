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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

  /* we do for-loops such that we could potentially leverage NUMA in future */
  int i1, i2, i3, i4, i5, i6;
  int N = buffer->N;
  int splits = buffer->splits;
  int fmb = buffer->fmb;
  int bfm = buffer->bfm;
  int H = buffer->H;
  int W = buffer->W;
#if defined(LIBXSMM_VLA)
  typedef element_type (*LIBXSMM_RESTRICT handle_data_type)[splits][fmb][H][W][bfm];
  typedef element_type (*LIBXSMM_RESTRICT user_data_type)[splits][fmb*bfm][H][W];
  const handle_data_type handle_data = (handle_data_type)buffer->data;
  const user_data_type user_data = (user_data_type)data;
#else
  element_type *const handle_data = (element_type*)buffer->data;
  const element_type *const user_data = (const element_type*)data;
  unsigned int hindexn[6], uindexn[5];
  unsigned int hshape[6], ushape[5];
  /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
  hshape[0] = bfm; hshape[1] = W; hshape[2] = H; hshape[3] = fmb; hshape[4] = splits; hshape[5] = N;
  ushape[0] = W; ushape[1] = H; ushape[2] = fmb * bfm; ushape[3] = splits; ushape[4] = N;
#endif
  for (i1 = 0; i1 < N; ++i1) {
    for (i2 = 0; i2 < splits; ++i2) {
      for (i3 = 0; i3 < fmb; ++i3) {
        for (i4 = 0; i4 < H; ++i4) {
          for (i5 = 0; i5 < W; ++i5) {
            for (i6 = 0; i6 < bfm; ++i6) {
#if defined(LIBXSMM_VLA)
              handle_data[i1][i2][i3][i4][i5][i6] = user_data[i1][i2][i3*bfm+i6][i4][i5];
#else
              size_t h, u;
              /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
              hindexn[0] = i6; hindexn[1] = i5; hindexn[2] = i4; hindexn[3] = i3; hindexn[4] = i2; hindexn[5] = i1;
              uindexn[0] = i5; uindexn[1] = i4; uindexn[2] = i3 * bfm + i6; uindexn[3] = i2; uindexn[4] = i1;
              LIBXSMM_CALC_INDEX1(size_t, h, 6, hindexn, hshape);
              LIBXSMM_CALC_INDEX1(size_t, u, 5, uindexn, ushape);
              handle_data[h] = user_data[u];
#endif
            }
          }
        }
      }
    }
  }
