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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

/* use for-loops to potentially leverage NUMA in the future */
int i1, i2, i3;
#if defined(LIBXSMM_DNN_COPY_LOW_PRECISION)
int lpb = tensor->layout->dim_size[0];
int bfm = tensor->layout->dim_size[1];
int fmb = tensor->layout->dim_size[2];
#else
int lpb = 1;
int bfm = tensor->layout->dim_size[0];
int fmb = tensor->layout->dim_size[1];
#endif

element_type* user_data = (element_type*)data;
LIBXSMM_VLA_DECL(3, const element_type, handle_data, (const element_type*)tensor->data, bfm, lpb);

for (i1 = 0; i1 < fmb; ++i1) {
  for (i2 = 0; i2 < bfm; ++i2) {
    for (i3 = 0; i3 < lpb; ++i3) {
      user_data[(i1*bfm*lpb) + (i2*lpb) + i3] = LIBXSMM_VLA_ACCESS(3, handle_data, i1, i2, i3, bfm, lpb);
    }
  }
}

