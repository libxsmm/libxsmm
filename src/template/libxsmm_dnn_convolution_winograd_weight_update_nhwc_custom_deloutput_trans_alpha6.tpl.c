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
LIBXSMM_VLA_DECL(4, float, input, inp, handle->ofwp, handle->blocksofm, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, tinp, ALPHA, handle->blocksofm*handle->cwino_upd.bimg, total_tiles, TDVLEN);
LIBXSMM_VLA_DECL(4, float, Ow, Owp, ALPHA, ALPHA, TDVLEN);
float I[ALPHA][ALPHA][TDVLEN];
unsigned int ti, tj;
int i, j, k;
int xdim, ydim;
const float rcp3 = 1.0f/3.0f;
const float rcp4  = 1.0f/4.0f;
const float rcp6  = 1.0f/6.0f;
const float rcp12 = 1.0f/12.0f;
const float rcp24 = 1.0f/24.0f;
float T[6][4][TDVLEN];
float t0[TDVLEN];
float t1[TDVLEN];
float t2[TDVLEN];
float t3[TDVLEN];
float t4[TDVLEN];

for (tj = 0; tj < handle->cwino_upd.jtiles; tj++) {
  for (ti = 0; ti < handle->cwino_upd.itiles; ti++) {
    for (j = 0; j < ALPHA; j++) {
      ydim = tj*(ALPHA - 2) + j;
      if (ydim < handle->ofh) {
        for (i = 0; i < ALPHA; i++) {
          xdim = ti*(ALPHA - 2) + i;
          if (xdim < handle->ofw) {
            LIBXSMM_PRAGMA_SIMD
            for (k = 0; k < TDVLEN; k++) {
              I[j][i][k] =
                LIBXSMM_VLA_ACCESS(4, input, ydim, xdim, 0, k, handle->ofwp, handle->blocksofm, TDVLEN);
            }
          } else {
            LIBXSMM_PRAGMA_SIMD
            for (k = 0; k < TDVLEN; k++) {
              I[j][i][k] = 0.0f;
            }
          }
        }
      } else {
        for (i = 0; i < ALPHA; i++) {
          LIBXSMM_PRAGMA_SIMD
          for (k = 0; k < TDVLEN; k++) {
            I[j][i][k] = 0.0f;
          }
        }
      }
    }
    /*trans_F_3x3_4x4(ALPHA, TDVLEN, Ow[tj*handle->cwino_upd.itiles + ti], I);*/

    /* inline code start */
    for (i = 0; i < 4; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (j = 0; j < TDVLEN; j++) {
        t0[j] = I[2][i][j] * rcp6;
        t1[j] = I[0][i][j] * -rcp6 - t0[j];
        t2[j] = I[0][i][j] * rcp24 + t0[j];
        t3[j] = (I[1][i][j]  + I[3][i][j]) * rcp6;
        t4[j] = I[1][i][j] * rcp12 + I[3][i][j] * rcp3;

        T[0][i][j] = I[0][i][j] * rcp4;
        T[1][i][j] = t1[j] - t3[j];
        T[2][i][j] = t1[j] + t3[j];
        T[3][i][j] = t2[j] + t4[j];
        T[4][i][j] = t2[j] - t4[j];
        T[5][i][j] = I[3][i][j];
      }
    }

    for (i = 0; i < 6; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (j = 0; j < TDVLEN; j++) {
        t0[j] = T[i][2][j] * rcp6;
        t1[j] = T[i][0][j] * -rcp6 - t0[j];
        t2[j] = T[i][0][j] * rcp24 + t0[j];
        t3[j] = (T[i][1][j] + T[i][3][j]) * rcp6;
        t4[j] = T[i][1][j] * rcp12 + T[i][3][j] * rcp3;

        LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_upd.itiles + ti, i, 0, j, ALPHA, ALPHA, TDVLEN) = T[i][0][j] * rcp4;
        LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_upd.itiles + ti, i, 1, j, ALPHA, ALPHA, TDVLEN) = t1[j] - t3[j];
        LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_upd.itiles + ti, i, 2, j, ALPHA, ALPHA, TDVLEN) = t1[j] + t3[j];
        LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_upd.itiles + ti, i, 3, j, ALPHA, ALPHA, TDVLEN) = t2[j] + t4[j];
        LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_upd.itiles + ti, i, 4, j, ALPHA, ALPHA, TDVLEN) = t2[j] - t4[j];
        LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_upd.itiles + ti, i, 5, j, ALPHA, ALPHA, TDVLEN) = T[i][3][j];
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
          LIBXSMM_VLA_ACCESS(5, output, j, i, 0, tj*handle->cwino_upd.itiles + ti, k, ALPHA, handle->blocksofm*handle->cwino_upd.bimg, total_tiles, TDVLEN) =
            LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_upd.itiles + ti, j, i, k, ALPHA, ALPHA, TDVLEN);
        }
      }
    }
  }
}

