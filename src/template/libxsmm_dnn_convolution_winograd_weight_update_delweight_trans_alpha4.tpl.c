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
  LIBXSMM_VLA_DECL(5, float, input, twp, ALPHA, (handle->blocksifm/VRATIO)*(handle->blocksofm/VRATIO), FDVLEN, FDVLEN);
  float Fw[ALPHA][ALPHA][FDVLEN][FDVLEN];
  float F[3][3][FDVLEN][FDVLEN];
  int r;
  int v;
  unsigned int i, j, k, l;
  float T[3][4][FDVLEN];
  float t0[FDVLEN];
  float M_[3][FDVLEN];

  for (j = 0; j < ALPHA; j++) {
    for (i = 0; i < ALPHA; i++) {
      for (v = 0; v < FDVLEN; v++) {
        LIBXSMM_PRAGMA_SIMD
        for (k = 0; k < FDVLEN; k++) {
          Fw[j][i][v][k] =
            LIBXSMM_VLA_ACCESS(5, input, j, i, 0, v, k, ALPHA, (handle->blocksifm/VRATIO)*(handle->blocksofm/VRATIO), FDVLEN, FDVLEN);
        }
      }
    }
  }
  /*trans_O_3x3_2x2(FDVLEN, Fw, F);*/

  /* inline code start */
  for (j = 0; j < FDVLEN; j++) {
    for (i = 0; i < 4; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < FDVLEN; k++) {
        t0[k] = Fw[1][i][j][k] + Fw[2][i][j][k];
        T[0][i][k] = t0[k] + Fw[0][i][j][k];
        T[1][i][k] = Fw[1][i][j][k] - Fw[2][i][j][k];
        T[2][i][k] = t0[k] + Fw[3][i][j][k];
      }
    }

    for (i = 0; i < 3; i++) {
      LIBXSMM_PRAGMA_SIMD
      for (k = 0; k < FDVLEN; k++) {
        t0[k] = T[i][1][k] + T[i][2][k];
        M_[0][k] = t0[k] + T[i][0][k];
        M_[1][k] = T[i][1][k] - T[i][2][k];
        M_[2][k] = t0[k] + T[i][3][k];

        for (l = 0; l < 3; l++) {
          F[i][l][j][k] = M_[l][k];
        }
      }
    }
  }
  /* inline code end */

  for (j = 0; j < 3; j++) {
    for (i = 0; i < 3; i++) {
      for (r = 0; r < VRATIO; r++) {
        for (v = 0; v < VRATIO; v++) {
          for (k = 0; k < TDVLEN; k++) {
            LIBXSMM_PRAGMA_SIMD
            for (l = 0; l < TDVLEN; l++) {
              LIBXSMM_VLA_ACCESS(6, output, v, r, j, i, k, l, handle->blocksifm, 3, 3, TDVLEN, TDVLEN) +=
                F[j][i][r*TDVLEN + k][v*TDVLEN + l];
            }
          }
        }
      }
    }
  }
