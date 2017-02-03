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
  LIBXSMM_VLA_DECL(4, float, input, inp, handle->ifhp, handle->ifwp, TDVLEN);
  LIBXSMM_VLA_DECL(5, float, output, tinp, ALPHA, (handle->blocksifm/VRATIO)*handle->cwino_upd.bimg, total_tiles, FDVLEN);
  float Iw[total_tiles][ALPHA][ALPHA][FDVLEN];
  float I[ALPHA][ALPHA][FDVLEN];
  unsigned int ti, tj;
  int i, j, k, r;
  int xdim, ydim;
  float T[6][6][FDVLEN];
  float t0[FDVLEN];
  float t1[FDVLEN];
  float t2[FDVLEN];
  float t3[FDVLEN];
  float t4[FDVLEN];
  float t5[FDVLEN];

  for (tj = 0; tj < handle->cwino_upd.jtiles; tj++) {
    for (ti = 0; ti < handle->cwino_upd.itiles; ti++) {
      for (j = 0; j < ALPHA; j++) {
        ydim = tj*(ALPHA - 2) + j - handle->desc.pad_h;
        if ((ydim < 0) || (ydim >= handle->desc.H)) {
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
            xdim = ti*(ALPHA - 2) + i - handle->desc.pad_w;
            if ((xdim < 0) || (xdim >= handle->desc.W)) {
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
                    LIBXSMM_VLA_ACCESS(4, input, r, ydim + handle->desc.pad_h_in, xdim + handle->desc.pad_w_in, k, handle->ifhp, handle->ifwp, TDVLEN);
                }
              }
            }
          }
        }
      }
      /*trans_I_3x3_4x4(ALPHA, FDVLEN, Iw[tj*handle->cwino_upd.itiles + ti], I);*/

      /* inline code start */
      for (i = 0; i < 6; i++) {
        LIBXSMM_PRAGMA_SIMD
        for (j = 0; j < FDVLEN; j++) {
          t0[j] = I[4][i][j] - 2.25f * I[2][i][j];
          t1[j] = I[3][i][j] - 2.25f * I[1][i][j];
          t2[j] = I[4][i][j] - 0.390625f * I[2][i][j];
          t3[j] = I[3][i][j] - 0.390625f * I[1][i][j];
          t4[j] = I[4][i][j] + 0.87890625f * I[0][i][j];
          t5[j] = I[5][i][j] + 0.87890625f * I[1][i][j];

          T[0][i][j] = t4[j] - 2.640625f * I[2][i][j];
          T[1][i][j] = t0[j] + 0.625f * t1[j];
          T[2][i][j] = t0[j] - 0.625f * t1[j];
          T[3][i][j] = t2[j] + 1.5f * t3[j];
          T[4][i][j] = t2[j] - 1.5f * t3[j];
          T[5][i][j] = t5[j] - 2.640625f * I[3][i][j];
        }
      }

      for (i = 0; i < 6; i++) {
        LIBXSMM_PRAGMA_SIMD
        for (j = 0; j < FDVLEN; j++) {
          t0[j] = T[i][4][j] - 2.25f * T[i][2][j];
          t1[j] = T[i][3][j] - 2.25f * T[i][1][j];
          t2[j] = T[i][4][j] - 0.390625f * T[i][2][j];
          t3[j] = T[i][3][j] - 0.390625f * T[i][1][j];
          t4[j] = T[i][4][j] + 0.87890625f * T[i][0][j];
          t5[j] = T[i][5][j] + 0.87890625f * T[i][1][j];

          Iw[tj*handle->cwino_upd.itiles + ti][i][0][j] = t4[j] - 2.640625f * T[i][2][j];
          Iw[tj*handle->cwino_upd.itiles + ti][i][1][j] = t0[j] + 0.625f * t1[j];
          Iw[tj*handle->cwino_upd.itiles + ti][i][2][j] = t0[j] - 0.625f * t1[j];
          Iw[tj*handle->cwino_upd.itiles + ti][i][3][j] = t2[j] + 1.5f * t3[j];
          Iw[tj*handle->cwino_upd.itiles + ti][i][4][j] = t2[j] - 1.5f * t3[j];
          Iw[tj*handle->cwino_upd.itiles + ti][i][5][j] = t5[j] - 2.640625f * T[i][3][j];
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
          for (k = 0; k < FDVLEN; k++) {
            LIBXSMM_VLA_ACCESS(5, output, j, i, 0, tj*handle->cwino_upd.itiles + ti, k, ALPHA, (handle->blocksifm/VRATIO)*handle->cwino_upd.bimg, total_tiles, FDVLEN) =
              Iw[tj*handle->cwino_upd.itiles + ti][j][i][k];
          }
        }
      }
    }
  }
