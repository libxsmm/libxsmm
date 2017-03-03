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

LIBXSMM_VLA_DECL(6, float, output, wp, handle->blocksifm, 3, 3, TDVLEN, TDVLEN);
LIBXSMM_VLA_DECL(5, float, input, twp, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN);
float Fw[ALPHA][ALPHA][TDVLEN][TDVLEN];
float F[3][3][TDVLEN][TDVLEN];
unsigned int i, j;
int k, l, v;
float T[3][6][TDVLEN];
float t0[TDVLEN];
float t1[TDVLEN];
float t2[TDVLEN];
float M_[3][TDVLEN];

for (j = 0; j < ALPHA; j++) {
  for (i = 0; i < ALPHA; i++) {
    for (v = 0; v < TDVLEN; v++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        Fw[j][i][v][k] =
          LIBXSMM_VLA_ACCESS(5, input, j, i, 0, v, k, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN);
      }
    }
  }
}
/*trans_O_3x3_4x4(TDVLEN, Fw, F);*/

/* inline code start */
for (j = 0; j < TDVLEN; j++) {
  for (i = 0; i < 6; i++) {
    LIBXSMM_PRAGMA_SIMD
    for (l = 0; l < TDVLEN; l++) {
      t0[l] = Fw[1][i][j][l] + Fw[2][i][j][l];
      t1[l] = Fw[3][i][j][l] + Fw[4][i][j][l];
      t2[l] = t1[l] * 4.0f  + Fw[5][i][j][l];

      T[0][i][l] = Fw[0][i][j][l] + t0[l] + t1[l];
      T[1][i][l] = (Fw[1][i][j][l] - Fw[2][i][j][l]) + 2.0f * (Fw[3][i][j][l] - Fw[4][i][j][l]);
      T[2][i][l] = t0[l] + t2[l];
    }
  }

  for (i = 0; i < 3; i++) {
    LIBXSMM_PRAGMA_SIMD
    for (l = 0; l < TDVLEN; l++) {
      t0[l] = T[i][1][l] + T[i][2][l];
      t1[l] = T[i][3][l] + T[i][4][l];
      t2[l] = t1[l] * 4.0f + T[i][5][l];

      M_[0][l] = T[i][0][l] + t0[l] + t1[l];
      M_[1][l] = (T[i][1][l] - T[i][2][l]) + 2.0f * (T[i][3][l] - T[i][4][l]);
      M_[2][l] = t0[l] + t2[l];

      for (k = 0; k < 3; k++) {
        F[i][k][j][l] = M_[k][l];
      }
    }
  }
}
/* inline code end */

for (j = 0; j < 3; j++) {
  for (i = 0; i < 3; i++) {
    for (k = 0; k < TDVLEN; k++) {
      LIBXSMM_PRAGMA_SIMD
      for (l = 0; l < TDVLEN; l++) {
        LIBXSMM_VLA_ACCESS(6, output, 0, 0, j, i, k, l, handle->blocksifm, 3, 3, TDVLEN, TDVLEN) +=
          F[j][i][k][l];
      }
    }
  }
}

