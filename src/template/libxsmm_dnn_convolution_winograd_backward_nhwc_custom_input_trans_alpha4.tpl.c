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
LIBXSMM_VLA_DECL(4, const float, input, inp, handle->ofwp, handle->blocksofm, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, tinp, ALPHA, (handle->blocksofm/VRATIO)*handle->cwino_bwd.bimg, total_tiles, FDVLEN);
LIBXSMM_VLA_DECL(4, float, Iw, Iwp, ALPHA, ALPHA, FDVLEN);
float I[ALPHA][ALPHA][FDVLEN];
unsigned int ti, tj;
int i, j, k, r;
int xdim, ydim;
const int l_pad = (handle->desc.W - handle->ofw)/2 + 1;
const int t_pad = (handle->desc.H - handle->ofh)/2 + 1;
float A0[FDVLEN];
float A1[FDVLEN];
float A2[FDVLEN];
float A3[FDVLEN];
float B0[FDVLEN];
float B1[FDVLEN];
float B2[FDVLEN];
float B3[FDVLEN];
float C0[FDVLEN];
float C1[FDVLEN];
float C2[FDVLEN];
float C3[FDVLEN];
float D0[FDVLEN];
float D1[FDVLEN];
float D2[FDVLEN];
float D3[FDVLEN];

for (tj = 0; tj < handle->cwino_bwd.jtiles; tj++) {
  for (ti = 0; ti < handle->cwino_bwd.itiles; ti++) {
    for (j = 0; j < ALPHA; j++) {
      ydim = tj*(ALPHA - 2) + j - t_pad;
      if ((ydim < 0) || (ydim >= handle->ofh)) {
        for (i = 0; i < ALPHA; i++) {
          for (r = 0; r < VRATIO; r++) {
            LIBXSMM_PRAGMA_SIMD
            for (k = 0; k < TDVLEN; k++) {
              I[j][i][r*TDVLEN + k] = 0.0f;
            }
          }
        }
      } else {
        for (i = 0; i < ALPHA; i++) {
          xdim = ti*(ALPHA - 2) + i - l_pad;
          if ((xdim < 0) || (xdim >= handle->ofw)) {
            for (r = 0; r < VRATIO; r++) {
              LIBXSMM_PRAGMA_SIMD
              for (k = 0; k < TDVLEN; k++) {
                I[j][i][r*TDVLEN + k] = 0.0f;
              }
            }
          } else {
            for (r = 0; r < VRATIO; r++) {
              LIBXSMM_PRAGMA_SIMD
              for (k = 0; k < TDVLEN; k++) {
                I[j][i][r*TDVLEN + k] =
                  LIBXSMM_VLA_ACCESS(4, input, ydim, xdim, r, k, handle->ofwp, handle->blocksofm, TDVLEN);
              }
            }
          }
        }
      }
    }
    /*trans_I_2x2_3x3(ALPHA, FDVLEN, Iw[tj*handle->cwino_bwd.itiles + ti], I);*/

    /* inline code start */
    for (i = 0; i < FDVLEN; i++) {
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
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 0, 0, i, ALPHA, ALPHA, FDVLEN) = A0[i] - A2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 0, 1, i, ALPHA, ALPHA, FDVLEN) = A1[i] + A2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 0, 2, i, ALPHA, ALPHA, FDVLEN) = A2[i] - A1[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 0, 3, i, ALPHA, ALPHA, FDVLEN) = A1[i] - A3[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 1, 0, i, ALPHA, ALPHA, FDVLEN) = B0[i] - B2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 1, 1, i, ALPHA, ALPHA, FDVLEN) = B1[i] + B2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 1, 2, i, ALPHA, ALPHA, FDVLEN) = B2[i] - B1[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 1, 3, i, ALPHA, ALPHA, FDVLEN) = B1[i] - B3[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 2, 0, i, ALPHA, ALPHA, FDVLEN) = C0[i] - C2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 2, 1, i, ALPHA, ALPHA, FDVLEN) = C1[i] + C2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 2, 2, i, ALPHA, ALPHA, FDVLEN) = C2[i] - C1[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 2, 3, i, ALPHA, ALPHA, FDVLEN) = C1[i] - C3[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 3, 0, i, ALPHA, ALPHA, FDVLEN) = D0[i] - D2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 3, 1, i, ALPHA, ALPHA, FDVLEN) = D1[i] + D2[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 3, 2, i, ALPHA, ALPHA, FDVLEN) = D2[i] - D1[i];
      LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, 3, 3, i, ALPHA, ALPHA, FDVLEN) = D1[i] - D3[i];
    }
    /* inline code end */

  }
}
for (j = 0; j < ALPHA; j++) {
  for (i = 0; i < ALPHA; i++) {
    for (tj = 0; tj < handle->cwino_bwd.jtiles; tj++) {
      for (ti = 0; ti < handle->cwino_bwd.itiles; ti++) {
        LIBXSMM_PRAGMA_SIMD
        for (k = 0; k < FDVLEN; k++) {
          LIBXSMM_VLA_ACCESS(5, output, j, i, 0, tj*handle->cwino_bwd.itiles + ti, k, ALPHA, (handle->blocksofm/VRATIO)*handle->cwino_bwd.bimg, total_tiles, FDVLEN) =
            LIBXSMM_VLA_ACCESS(4, Iw, tj*handle->cwino_bwd.itiles + ti, j, i, k, ALPHA, ALPHA, FDVLEN);
        }
      }
    }
  }
}

