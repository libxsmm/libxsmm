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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

int total_tiles = handle->cwino_upd.itiles*handle->cwino_upd.jtiles;
LIBXSMM_VLA_DECL(4, float, input, inp, handle->ifwp, handle->blocksifm, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, tinp, ALPHA, handle->blocksifm*handle->cwino_upd.bimg, total_tiles, TDVLEN);
LIBXSMM_VLA_DECL(4, float, Iw, Iwp, ALPHA, ALPHA, TDVLEN);
float I[ALPHA][ALPHA][TDVLEN];
unsigned int ti, tj;
int i, j, k;
int xdim, ydim;
float T[4][4][TDVLEN];

for (tj = 0; tj < handle->cwino_upd.jtiles; tj++) {
  for (ti = 0; ti < handle->cwino_upd.itiles; ti++) {
    for (j = 0; j < ALPHA; j++) {
      ydim = tj*(ALPHA - 2) + j - handle->desc.pad_h;
      if ((ydim < 0) || (ydim >= handle->desc.H)) {
        for (i = 0; i < ALPHA; i++) {
          LIBXSMM_PRAGMA_SIMD
          for (k = 0; k < TDVLEN; k++) {
            I[j][i][k] = 0.0f;
          }
        }
      } else {
        for (i = 0; i < ALPHA; i++) {
          xdim = ti*(ALPHA - 2) + i - handle->desc.pad_w;
          if ((xdim < 0) || (xdim >= handle->desc.W)) {
            LIBXSMM_PRAGMA_SIMD
            for (k = 0; k < TDVLEN; k++) {
              I[j][i][k] = 0.0f;
            }
          } else {
            LIBXSMM_PRAGMA_SIMD
            for (k = 0; k < TDVLEN; k++) {
              I[j][i][k] =
                LIBXSMM_VLA_ACCESS(4, input, ydim + handle->desc.pad_h_in, xdim + handle->desc.pad_w_in, 0, k, handle->ifwp, handle->blocksifm, TDVLEN);
            }
          }
        }
      }
    }
    /*trans_I_3x3_2x2(ALPHA, TDVLEN, Iw[tj*handle->cwino_upd.itiles + ti], I);*/

    /* inline code start */
    for (i = 0; i < 4; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (j = 0; j < TDVLEN; j++) {
        T[0][i][j] = I[0][i][j] - I[2][i][j];
        T[1][i][j] = I[1][i][j] + I[2][i][j];
        T[2][i][j] = I[2][i][j] - I[1][i][j];
        T[3][i][j] = I[3][i][j] - I[1][i][j];
      }
    }

    for (i = 0; i < 4; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (j = 0; j < TDVLEN; j++) {
        LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_upd.itiles + ti, i, 0, j, ALPHA, ALPHA, TDVLEN) = T[i][0][j] - T[i][2][j];
        LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_upd.itiles + ti, i, 1, j, ALPHA, ALPHA, TDVLEN) = T[i][1][j] + T[i][2][j];
        LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_upd.itiles + ti, i, 2, j, ALPHA, ALPHA, TDVLEN) = T[i][2][j] - T[i][1][j];
        LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_upd.itiles + ti, i, 3, j, ALPHA, ALPHA, TDVLEN) = T[i][3][j] - T[i][1][j];
      }
    }
    /* inline code end */

  }
}
for (j = 0; j < ALPHA; j++) {
  for (i = 0; i < ALPHA; i++) {
    for (tj = 0; tj < handle->cwino_upd.jtiles; tj++) {
      for (ti = 0; ti < handle->cwino_upd.itiles; ti++) {
        LIBXSMM_PRAGMA_SIMD
        for (k = 0; k < TDVLEN; k++) {
          LIBXSMM_VLA_ACCESS(5, output, j, i, 0, tj*handle->cwino_upd.itiles + ti, k, ALPHA, handle->blocksifm*handle->cwino_upd.bimg, total_tiles, TDVLEN) =
            LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_upd.itiles + ti, j, i, k, ALPHA, ALPHA, TDVLEN);
        }
      }
    }
  }
}

