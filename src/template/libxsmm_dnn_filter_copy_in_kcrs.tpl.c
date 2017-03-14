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
/* Alexander Heinecke, Evangelos Georganas, Hans Pabst (Intel Corp.)
******************************************************************************/

/* @TODO: use for-loops to potentially leverage NUMA in the future */
int i1, i2, i3, i4, i5, i6, i7;
int ifmb = filter->ifmb;
int bifm = filter->bifm;
int ofmb = filter->ofmb;
int bofm = filter->bofm;
int R = filter->R;
int S = filter->S;
int lpb = filter->lpb;
int C = ifmb * bifm * lpb;
LIBXSMM_VLA_DECL(7, element_type, handle_data_1, (element_type*)filter->data, ifmb, R, S, bifm, bofm, lpb);
LIBXSMM_VLA_DECL(6, element_type, handle_data_2, (element_type*)filter->data, ifmb, R, S, bifm, bofm);
LIBXSMM_VLA_DECL(4, const element_type, user_data, (const element_type*)data, ifmb * bifm * lpb, R, S);

if (filter->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1) {
  for (i1 = 0; i1 < ofmb; ++i1) {
    for (i2 = 0; i2 < ifmb; ++i2) {
      for (i3 = 0; i3 < R; ++i3) {
        for (i4 = 0; i4 < S; ++i4) {
          for (i5 = 0; i5 < bifm; ++i5) {
            for (i6 = 0; i6 < bofm; ++i6) {
              for (i7 = 0; i7 < lpb; ++i7) {
                LIBXSMM_VLA_ACCESS(7, handle_data_1, i1, i2, i3, i4, i5, i6, i7, ifmb, R, S, bifm, bofm, lpb) =
                LIBXSMM_VLA_ACCESS(4, user_data, i1 * bofm + i6, (i2*bifm*lpb) + (i5*lpb) + i7, i3, i4, ifmb * bifm * lpb, R, S);
              }
            }
          }
        }
      }
    }
  }
} else if (filter->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
  for ( i1 = 0; i1 < ofmb; i1++ ) {
    for ( i2 = 0; i2 < ifmb; i2++ ) {
      for ( i3 = 0; i3 < R; i3++ ) {
        for ( i4 = 0; i4 < S; i4++ ) {
          for ( i5 = 0; i5 < bifm; i5++ ) {
            for ( i6 = 0; i6 < bofm; i6++ ) {
              LIBXSMM_VLA_ACCESS(6, handle_data_2, i1, i2, i3, i4, i5, i6, ifmb, R, S, bifm, bofm) =
              LIBXSMM_VLA_ACCESS(4, user_data, (i1*bofm)+i6, (i2*bifm)+i5, i3, i4, C, R, S);
            }
          }
        }
      }
    }
  }
}

