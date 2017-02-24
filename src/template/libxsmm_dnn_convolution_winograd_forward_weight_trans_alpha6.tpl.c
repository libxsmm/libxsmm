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

LIBXSMM_VLA_DECL(6, float, input, wp, handle->blocksifm, 3, 3, TDVLEN, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, twp, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN);
float Fw[ALPHA][ALPHA][TDVLEN][TDVLEN];
float F[3][3][TDVLEN][TDVLEN];
unsigned int i, j;
int k, l, v;
const float rcp4  = 1.0f/4.0f;
const float rcp6  = 1.0f/6.0f;
const float rcp12 = 1.0f/12.0f;
const float rcp24 = 1.0f/24.0f;
float T[6][3][TDVLEN];
float Fw_[6][TDVLEN];
float t0[TDVLEN];
float t1[TDVLEN];
float t2[TDVLEN];

for (j = 0; j < 3; j++) {
  for (i = 0; i < 3; i++) {
    for (k = 0; k < TDVLEN; k++) {
      LIBXSMM_PRAGMA_SIMD
      for (l = 0; l < TDVLEN; l++) {
        F[j][i][k][l] =
          LIBXSMM_VLA_ACCESS(6, input, 0, 0, j, i, k, l, handle->blocksifm, 3, 3, TDVLEN, TDVLEN);
      }
    }
  }
}
/*trans_F_4x4_3x3(TDVLEN, Fw, F);*/

/* inline code start */
for (j = 0; j < TDVLEN; j++) {
  for (i = 0; i < 3; i++) {
    LIBXSMM_PRAGMA_SIMD
    for (k = 0; k < TDVLEN; k++) {
      t0[k] = rcp6 * F[2][i][j][k];
      t1[k] = -t0[k] - rcp6*F[0][i][j][k];
      t2[k] = t0[k] + rcp24*F[0][i][j][k];
      T[0][i][k] = rcp4 * F[0][i][j][k];
      T[1][i][k] = t1[k] - rcp6*F[1][i][j][k];
      T[2][i][k] = t1[k] + rcp6*F[1][i][j][k];
      T[3][i][k] = t2[k] + rcp12*F[1][i][j][k];
      T[4][i][k] = t2[k] - rcp12*F[1][i][j][k];
      T[5][i][k] = F[2][i][j][k];
    }
  }
  for (i = 0; i < 6; i++) {
    LIBXSMM_PRAGMA_SIMD
    for (k = 0; k < TDVLEN; k++) {
      t0[k] = rcp6 * T[i][2][k];
      t1[k] = -t0[k] - rcp6*T[i][0][k];
      t2[k] = t0[k] + rcp24*T[i][0][k];
      Fw_[0][k] = rcp4 * T[i][0][k];
      Fw_[1][k] = t1[k] - rcp6*T[i][1][k];
      Fw_[2][k] = t1[k] + rcp6*T[i][1][k];
      Fw_[3][k] = t2[k] + rcp12*T[i][1][k];
      Fw_[4][k] = t2[k] - rcp12*T[i][1][k];
      Fw_[5][k] = T[i][2][k];

      for (l = 0; l < 6; l++) {
        Fw[i][l][j][k] = Fw_[l][k];
      }
    }
  }
}
/* inline code end */

for (j = 0; j < ALPHA; j++) {
  for (i = 0; i < ALPHA; i++) {
    for (v = 0; v < TDVLEN; v++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        LIBXSMM_VLA_ACCESS(5, output, j, i, 0, v, k, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN) =
          Fw[j][i][v][k];
      }
    }
  }
}

