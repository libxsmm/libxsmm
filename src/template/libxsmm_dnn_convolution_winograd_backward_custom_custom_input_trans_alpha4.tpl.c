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

int total_tiles = handle->cwino_bwd.itiles*handle->cwino_bwd.jtiles;
LIBXSMM_VLA_DECL(4, const float, input, inp, handle->ofhp, handle->ofwp, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, tinp, ALPHA, handle->blocksofm*handle->cwino_bwd.bimg, total_tiles, TDVLEN);
LIBXSMM_VLA_DECL(4, float, Iw, Iwp, ALPHA, ALPHA, TDVLEN);
float I[ALPHA][ALPHA][TDVLEN];
unsigned int ti, tj;
int i, j, k;
int xdim, ydim;
const int l_pad = (handle->desc.W - handle->ofw)/2 + 1;
const int t_pad = (handle->desc.H - handle->ofh)/2 + 1;
float A0[TDVLEN];
float A1[TDVLEN];
float A2[TDVLEN];
float A3[TDVLEN];
float B0[TDVLEN];
float B1[TDVLEN];
float B2[TDVLEN];
float B3[TDVLEN];
float C0[TDVLEN];
float C1[TDVLEN];
float C2[TDVLEN];
float C3[TDVLEN];
float D0[TDVLEN];
float D1[TDVLEN];
float D2[TDVLEN];
float D3[TDVLEN];

for (tj = 0; tj < handle->cwino_bwd.jtiles; tj++) {
  for (ti = 0; ti < handle->cwino_bwd.itiles; ti++) {
    for (j = 0; j < ALPHA; j++) {
      ydim = tj*(ALPHA - 2) + j - t_pad;
      if ((ydim < 0) || (ydim >= handle->ofh)) {
        for (i = 0; i < ALPHA; i++) {
          LIBXSMM_PRAGMA_SIMD
          for (k = 0; k < TDVLEN; k++) {
            I[j][i][k] = 0.0f;
          }
        }
      } else {
        for (i = 0; i < ALPHA; i++) {
          xdim = ti*(ALPHA - 2) + i - l_pad;
          if ((xdim < 0) || (xdim >= handle->ofw)) {
            LIBXSMM_PRAGMA_SIMD
            for (k = 0; k < TDVLEN; k++) {
              I[j][i][k] = 0.0f;
            }
          } else {
            LIBXSMM_PRAGMA_SIMD
            for (k = 0; k < TDVLEN; k++) {
              I[j][i][k] =
                LIBXSMM_VLA_ACCESS(4, input, 0, ydim, xdim, k, handle->ofhp, handle->ofwp, TDVLEN);
            }
          }
        }
      }
    }
    /*trans_I_2x2_3x3(ALPHA, TDVLEN, Iw[tj*handle->cwino_bwd.itiles + ti], I);*/

    /* inline code start */
    for (i = 0; i < TDVLEN; i++) {
      A0[i] = I[0][0][i] - I[2][0][i];
      A1[i] = I[0][1][i] - I[2][1][i];
      A2[i] = I[0][2][i] - I[2][2][i];
      A3[i] = I[0][3][i] - I[2][3][i];
      B0[i] = I[1][0][i] + I[2][0][i];
      B1[i] = I[1][1][i] + I[2][1][i];
      B2[i] = I[1][2][i] + I[2][2][i];
      B3[i] = I[1][3][i] + I[2][3][i];
      C0[i] = I[2][0][i] - I[1][0][i];
      C1[i] = I[2][1][i] - I[1][1][i];
      C2[i] = I[2][2][i] - I[1][2][i];
      C3[i] = I[2][3][i] - I[1][3][i];
      D0[i] = I[1][0][i] - I[3][0][i];
      D1[i] = I[1][1][i] - I[3][1][i];
      D2[i] = I[1][2][i] - I[3][2][i];
      D3[i] = I[1][3][i] - I[3][3][i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 0, 0, i, ALPHA, ALPHA, TDVLEN) = A0[i] - A2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 0, 1, i, ALPHA, ALPHA, TDVLEN) = A1[i] + A2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 0, 2, i, ALPHA, ALPHA, TDVLEN) = A2[i] - A1[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 0, 3, i, ALPHA, ALPHA, TDVLEN) = A1[i] - A3[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 1, 0, i, ALPHA, ALPHA, TDVLEN) = B0[i] - B2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 1, 1, i, ALPHA, ALPHA, TDVLEN) = B1[i] + B2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 1, 2, i, ALPHA, ALPHA, TDVLEN) = B2[i] - B1[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 1, 3, i, ALPHA, ALPHA, TDVLEN) = B1[i] - B3[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 2, 0, i, ALPHA, ALPHA, TDVLEN) = C0[i] - C2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 2, 1, i, ALPHA, ALPHA, TDVLEN) = C1[i] + C2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 2, 2, i, ALPHA, ALPHA, TDVLEN) = C2[i] - C1[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 2, 3, i, ALPHA, ALPHA, TDVLEN) = C1[i] - C3[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 3, 0, i, ALPHA, ALPHA, TDVLEN) = D0[i] - D2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 3, 1, i, ALPHA, ALPHA, TDVLEN) = D1[i] + D2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 3, 2, i, ALPHA, ALPHA, TDVLEN) = D2[i] - D1[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 3, 3, i, ALPHA, ALPHA, TDVLEN) = D1[i] - D3[i];
    }
    /* inline code end */
  }
}
for (j = 0; j < ALPHA; j++) {
  for (i = 0; i < ALPHA; i++) {
    for (tj = 0; tj < handle->cwino_bwd.jtiles; tj++) {
      for (ti = 0; ti < handle->cwino_bwd.itiles; ti++) {
        LIBXSMM_PRAGMA_SIMD
        for (k = 0; k < TDVLEN; k++) {
          LIBXSMM_VLA_ACCESS(5, output, j, i, 0, tj*handle->cwino_bwd.itiles + ti, k, ALPHA, handle->blocksofm*handle->cwino_bwd.bimg, total_tiles, TDVLEN) =
            LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, j, i, k, ALPHA, ALPHA, TDVLEN);
        }
      }
    }
  }
}

