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

  int i1, i2, i3, i4, i5, i6, i7;
  int splits = filter->splits;
  int ifmb = filter->ifmb;
  int bifm = filter->bifm;
  int ofmb = filter->ofmb;
  int bofm = filter->bofm;
  int R = filter->R;
  int S = filter->S;
#if defined(LIBXSMM_VLA)
  typedef element_type (*LIBXSMM_RESTRICT handle_data_type)[ofmb][ifmb][R][S][bifm][bofm];
  typedef element_type (*LIBXSMM_RESTRICT user_data_type)[ofmb*bofm][ifmb*bifm][R][S];
  const handle_data_type handle_data = (handle_data_type)filter->data;
  const user_data_type user_data = (user_data_type)data;
#else
  element_type *const handle_data = (element_type*)filter->data;
  const element_type *const user_data = (const element_type*)data;
  unsigned int hindexn[7], uindexn[5];
  unsigned int hshape[7], ushape[5];
/* arrays must be initialized separately to avoid warning about values not computable at init.-time */
  hshape[0] = bofm; hshape[1] = bifm; hshape[2] = S; hshape[3] = R; hshape[4] = ifmb; hshape[5] = ofmb; hshape[6] = splits;
  ushape[0] = S; ushape[1] = R; ushape[2] = ifmb * bifm; ushape[3] = ofmb * bofm; ushape[4] = splits;
#endif
  for (i1 = 0; i1 < splits; ++i1) {
    for (i2 = 0; i2 < ofmb; ++i2) {
      for (i3 = 0; i3 < ifmb; ++i3) {
        for (i4 = 0; i4 < R; ++i4) {
          for (i5 = 0; i5 < S; ++i5) {
            for (i6 = 0; i6 < bifm; ++i6) {
              for (i7 = 0; i7 < bofm; ++i7) {
#if defined(LIBXSMM_VLA)
                handle_data[i1][i2][i3][i4][i5][i6][i7] = user_data[i1][i2*bofm+i7][i3*bifm+i6][i4][i5];
#else
                size_t h, u;
                /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
                hindexn[0] = i7; hindexn[1] = i6; hindexn[2] = i5; hindexn[3] = i4; hindexn[4] = i3; hindexn[5] = i2; hindexn[6] = i1;
                uindexn[0] = i5; uindexn[1] = i4; uindexn[2] = i3 * bifm + i6; uindexn[3] = i2 * bofm + i7; uindexn[4] = i1;
                LIBXSMM_CALC_INDEX1(size_t, h, 7, hindexn, hshape);
                LIBXSMM_CALC_INDEX1(size_t, u, 5, uindexn, ushape);
                handle_data[h] = user_data[u];
#endif
              }
            }
          }
        }
      }
    }
  }
