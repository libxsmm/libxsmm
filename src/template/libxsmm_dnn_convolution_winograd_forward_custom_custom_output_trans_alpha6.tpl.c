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

const int total_tiles = handle->cwino_fwd.itiles*handle->cwino_fwd.jtiles;
LIBXSMM_VLA_DECL(5, const float, input, toutp, ALPHA, handle->blocksofm*handle->cwino_fwd.bimg, total_tiles, TDVLEN);
LIBXSMM_VLA_DECL(4, float, output, outp, handle->ofhp, handle->ofwp, TDVLEN);
LIBXSMM_VLA_DECL(4, float, Ow, Owp, ALPHA, ALPHA, TDVLEN);
float O[ALPHA-2][ALPHA-2][TDVLEN];
unsigned int ti, tj;
int i, j, k;
int xdim, ydim;
float T[4][6][TDVLEN];
float t0[TDVLEN];
float t1[TDVLEN];
float t2[TDVLEN];
float t3[TDVLEN];

for (j = 0; j < ALPHA; j++) {
  for (i = 0; i < ALPHA; i++) {
    for (tj = 0; tj < handle->cwino_fwd.jtiles; tj++) {
      for (ti = 0; ti < handle->cwino_fwd.itiles; ti++) {
        LIBXSMM_PRAGMA_SIMD
        for (k = 0; k < TDVLEN; k++) {
          LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, j, i, k, ALPHA, ALPHA, TDVLEN) =
            LIBXSMM_VLA_ACCESS(5, input, j, i, 0, tj*handle->cwino_fwd.itiles + ti, k, ALPHA, handle->blocksofm*handle->cwino_fwd.bimg, total_tiles, TDVLEN);
        }
      }
    }
  }
}
for (tj = 0; tj < handle->cwino_fwd.jtiles; tj++) {
  for (ti = 0; ti < handle->cwino_fwd.itiles; ti++) {
    /*trans_O_4x4_3x3(ALPHA-2, TDVLEN, Ow[tj*handle->cwino_fwd.itiles + ti], O);*/

    /* inline code start */
    for (i = 0; i < 6; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (j = 0; j < TDVLEN; j++) {
        t0[j] = LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 1, i, j, ALPHA, ALPHA, TDVLEN) + LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 2, i, j, ALPHA, ALPHA, TDVLEN);
        t1[j] = LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 3, i, j, ALPHA, ALPHA, TDVLEN) + LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 4, i, j, ALPHA, ALPHA, TDVLEN);
        t2[j] = LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 1, i, j, ALPHA, ALPHA, TDVLEN) - LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 2, i, j, ALPHA, ALPHA, TDVLEN);
        t3[j] = LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 3, i, j, ALPHA, ALPHA, TDVLEN) - LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 4, i, j, ALPHA, ALPHA, TDVLEN);

        T[0][i][j] = t0[j] + t1[j]     + LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 0, i, j, ALPHA, ALPHA, TDVLEN);
        T[1][i][j] = t2[j] + t3[j]*2.f;
        T[2][i][j] = t0[j] + t1[j]*4.f;
        T[3][i][j] = t2[j] + t3[j]*8.f + LIBXSMM_VLA_ACCESS(4, Ow, tj*handle->cwino_fwd.itiles + ti, 5, i, j, ALPHA, ALPHA, TDVLEN);
      }
    }

    for (i = 0; i < 4; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (j = 0; j < TDVLEN; j++) {
        t0[j] = T[i][1][j] + T[i][2][j];
        t1[j] = T[i][3][j] + T[i][4][j];
        t2[j] = T[i][1][j] - T[i][2][j];
        t3[j] = T[i][3][j] - T[i][4][j];

        O[i][0][j] = t0[j] + t1[j]     + T[i][0][j];
        O[i][1][j] = t2[j] + t3[j]*2.f;
        O[i][2][j] = t0[j] + t1[j]*4.f;
        O[i][3][j] = t2[j] + t3[j]*8.f + T[i][5][j];
      }
    }
    /* inline code end */

    for (j = 0; j < ALPHA-2; j++) {
      ydim = tj*(ALPHA - 2) + j;
      if (ydim < handle->ofh) {
        for (i = 0; i < ALPHA-2; i++) {
          xdim = ti*(ALPHA - 2) + i;
          if (xdim < handle->ofw) {
            LIBXSMM_PRAGMA_SIMD
            for (k = 0; k < TDVLEN; k++) {
              LIBXSMM_VLA_ACCESS(4, output, 0, ydim, xdim, k, handle->ofhp, handle->ofwp, TDVLEN) +=
                O[j][i][k]; /* + bias[0][k]; */
            }
          }
        }
      }
    }
  }
}

