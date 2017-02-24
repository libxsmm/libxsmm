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

LIBXSMM_VLA_DECL(6, const float, input, wp, handle->blocksifm, 3, 3, TDVLEN, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, twp, ALPHA, handle->blocksifm*handle->blocksofm, TDVLEN, TDVLEN);
float Fw[ALPHA][ALPHA][TDVLEN][TDVLEN];
float F[3][3][TDVLEN][TDVLEN];
unsigned int i, j, k, l;
int v;
float Fw_[4][4][TDVLEN];
float x0[TDVLEN];
float x1[TDVLEN];
float x2[TDVLEN];
float x3[TDVLEN];
float x4[TDVLEN];
float x5[TDVLEN];
float x6[TDVLEN];
float x7[TDVLEN];
float x8[TDVLEN];
float x9[TDVLEN];
float x10[TDVLEN];
float x11[TDVLEN];
float x12[TDVLEN];
float x13[TDVLEN];
float x14[TDVLEN];
float x15[TDVLEN];
const float half    = 0.5f;
const float quarter = 0.25f;

for (j = 0; j < 3; j++) {
  for (i = 0; i < 3; i++) {
    for (v = 0; v < TDVLEN; v++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < TDVLEN; k++) {
        F[j][i][k][v] =
          LIBXSMM_VLA_ACCESS(6, input, 0, 0, 2-j, 2-i, v, k, handle->blocksifm, 3, 3, TDVLEN, TDVLEN);
      }
    }
  }
}
/*trans_F_2x2_3x3(TDVLEN, Fw, F);*/

/* inline code start */
for (i = 0; i < TDVLEN; i++) {
  LIBXSMM_PRAGMA_SIMD
  for (l = 0; l < TDVLEN; l++) {
    x0[l]  = half*F[0][1][i][l];
    x1[l]  = F[0][0][i][l] + F[0][2][i][l];
    x2[l]  = F[2][0][i][l] + F[2][2][i][l];
    x8[l]  = half*F[2][1][i][l];
    x10[l] = x1[l] + x2[l];
    x5[l]  = F[0][1][i][l] + F[2][1][i][l];
    x7[l]  = F[1][0][i][l] + F[1][2][i][l];
    x9[l]  = quarter*F[1][1][i][l];
    x11[l] = x10[l] + x5[l];
    x14[l] = x10[l] - x5[l];
    x13[l] = quarter*x7[l] + x9[l];
    x15[l] = quarter*x7[l] - x9[l];
    x3[l]  = half*F[1][0][i][l];
    x4[l]  = half*F[1][2][i][l];
    x6[l]  = F[0][0][i][l] + F[2][0][i][l];
    x12[l] = F[0][2][i][l] + F[2][2][i][l];
    Fw_[0][1][l] = half*x1[l] + x0[l];
    Fw_[0][2][l] = half*x1[l] - x0[l];
    Fw_[0][0][l] = F[0][0][i][l];
    Fw_[0][3][l] = F[0][2][i][l];
    Fw_[3][0][l] = F[2][0][i][l];
    Fw_[3][1][l] = half*x2[l] + x8[l];
    Fw_[3][2][l] = half*x2[l] - x8[l];
    Fw_[3][3][l] = F[2][2][i][l];
    Fw_[1][1][l] = quarter*x11[l] + x13[l];
    Fw_[2][1][l] = quarter*x11[l] - x13[l];
    Fw_[1][2][l] = quarter*x14[l] + x15[l];
    Fw_[2][2][l] = quarter*x14[l] - x15[l];
    Fw_[1][0][l] = half*x6[l] + x3[l];
    Fw_[1][3][l] = half*x12[l] + x4[l];
    Fw_[2][0][l] = half*x6[l] - x3[l];
    Fw_[2][3][l] = half*x12[l] - x4[l];
    LIBXSMM_PRAGMA_UNROLL
    for (k = 0; k < 4; k++) {
      LIBXSMM_PRAGMA_UNROLL
      for (j = 0; j < 4; j++) {
        Fw[k][j][i][l] = Fw_[k][j][l];
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

