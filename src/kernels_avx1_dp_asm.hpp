/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
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

void avx1_kernel_12xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast, int max_local_N) {
  if (call != -1) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
//    codestream << "                         \"vbroadcastsd " << 8 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
//    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
//    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
//    codestream << "                         \"vbroadcastsd (%%r8), %%ymm0\\n\\t\"" << std::endl;
//    codestream << "                         \"vbroadcastsd " << (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
//    codestream << "                         \"vbroadcastsd " << (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    codestream << "                         \"vmulpd %%ymm3, %%ymm" << l_n << ", %%ymm" << 4 + l_n << "\\n\\t\"" << std::endl;
    codestream << "                         \"vaddpd %%ymm" << 4 + l_n << ", %%ymm" << 7 + (3*l_n) << ", %%ymm" << 7 + (3*l_n) << "\\n\\t\"" << std::endl;
  }
//  codestream << "                         \"vmulpd %%ymm3, %%ymm0, %%ymm4\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm4, %%ymm7, %%ymm7\\n\\t\"" << std::endl;
//  codestream << "                         \"vmulpd %%ymm3, %%ymm1, %%ymm5\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm5, %%ymm10, %%ymm10\\n\\t\"" << std::endl;
//  codestream << "                         \"vmulpd %%ymm3, %%ymm2, %%ymm6\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm6, %%ymm13, %%ymm13\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"vmovapd 32(%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd 32(%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    codestream << "                         \"vmulpd %%ymm3, %%ymm" << l_n << ", %%ymm" << 4 + l_n << "\\n\\t\"" << std::endl;
    codestream << "                         \"vaddpd %%ymm" << 4 + l_n << ", %%ymm" << 8 + (3*l_n) << ", %%ymm" << 8 + (3*l_n) << "\\n\\t\"" << std::endl;
  }
//  codestream << "                         \"vmulpd %%ymm3, %%ymm0, %%ymm4\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm4, %%ymm8, %%ymm8\\n\\t\"" << std::endl;
//  codestream << "                         \"vmulpd %%ymm3, %%ymm1, %%ymm5\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm5, %%ymm11, %%ymm11\\n\\t\"" << std::endl;
//  codestream << "                         \"vmulpd %%ymm3, %%ymm2, %%ymm6\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm6, %%ymm14, %%ymm14\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"vmovapd 64(%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd 64(%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    codestream << "                         \"vmulpd %%ymm3, %%ymm" << l_n << ", %%ymm" << 4 + l_n << "\\n\\t\"" << std::endl;
    codestream << "                         \"vaddpd %%ymm" << 4 + l_n << ", %%ymm" << 9 + (3*l_n) << ", %%ymm" << 9 + (3*l_n) << "\\n\\t\"" << std::endl;
    if (l_n == 0) {
      codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
    }
  }
//  codestream << "                         \"vmulpd %%ymm3, %%ymm0, %%ymm4\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm4, %%ymm9, %%ymm9\\n\\t\"" << std::endl;
//  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
//  codestream << "                         \"vmulpd %%ymm3, %%ymm1, %%ymm5\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm5, %%ymm12, %%ymm12\\n\\t\"" << std::endl;
//  codestream << "                         \"vmulpd %%ymm3, %%ymm2, %%ymm6\\n\\t\"" << std::endl;
//  codestream << "                         \"vaddpd %%ymm6, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
}

void avx1_kernel_8xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast, int max_local_N) {
  if (call != -1) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    codestream << "                         \"vmulpd %%ymm3, %%ymm" << l_n << ", %%ymm" << 5 + l_n << "\\n\\t\"" << std::endl;
    codestream << "                         \"vaddpd %%ymm" << 5 + l_n << ", %%ymm" << 10 + (2*l_n) << ", %%ymm" << 10 + (2*l_n) << "\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"vmovapd 32(%%r9), %%ymm4\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd 32(%%r9), %%ymm4\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    if (l_n == 0) {
      codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
    }
    codestream << "                         \"vmulpd %%ymm4, %%ymm" << l_n << ", %%ymm" << 5 + l_n << "\\n\\t\"" << std::endl;
    codestream << "                         \"vaddpd %%ymm" << 5 + l_n << ", %%ymm" << 11 + (2*l_n) << ", %%ymm" << 11 + (2*l_n) << "\\n\\t\"" << std::endl;
  }

#if 0
  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%ymm4\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd 32(%%r9), %%ymm5\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%ymm4\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd 32(%%r9), %%ymm5\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << 8 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd (%%r8), %%ymm0\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%ymm0, %%ymm4, %%ymm9\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm9, %%ymm10, %%ymm10\\n\\t\"" << std::endl;
  codestream << "                         \"vmulpd %%ymm0, %%ymm5, %%ymm8\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm8, %%ymm11, %%ymm11\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%ymm1, %%ymm4, %%ymm9\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm9, %%ymm12, %%ymm12\\n\\t\"" << std::endl;
  codestream << "                         \"vmulpd %%ymm1, %%ymm5, %%ymm8\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm8, %%ymm13, %%ymm13\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%ymm2, %%ymm4, %%ymm9\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm9, %%ymm14, %%ymm14\\n\\t\"" << std::endl;
  codestream << "                         \"vmulpd %%ymm2, %%ymm5, %%ymm8\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm8, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
#endif
}

void avx1_kernel_4xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast, int max_local_N) {
  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    if (l_n == 0) {
      codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
    }
    if (call != -1) {
      codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vbroadcastsd " << (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
      if (l_n == (max_local_N - 1)) {
        codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
      }
    }
    codestream << "                         \"vmulpd %%ymm3, %%ymm" << l_n << ", %%ymm" << 4 + l_n << "\\n\\t\"" << std::endl;
    codestream << "                         \"vaddpd %%ymm" << 4 + l_n << ", %%ymm" << 13 + l_n << ", %%ymm" << 13 + l_n << "\\n\\t\"" << std::endl;
  }
#if 0
  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%ymm4\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%ymm4\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << 8 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd (%%r8), %%ymm0\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%ymm0, %%ymm4, %%ymm12\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm12, %%ymm13, %%ymm13\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%ymm1, %%ymm4, %%ymm11\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm11, %%ymm14, %%ymm14\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%ymm2, %%ymm4, %%ymm10\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%ymm10, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
#endif
}

void avx1_kernel_2xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast, int max_local_N) {
  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    if (l_n == 0) {
      codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
    }
    if (call != -1) {
      codestream << "                         \"movddup " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%xmm" << l_n << "\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"movddup " << (ldb * l_n * 8) << "(%%r8), %%xmm" << l_n << "\\n\\t\"" << std::endl;
      if (l_n == (max_local_N - 1)) {
        codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
      }
    }
    codestream << "                         \"vmulpd %%xmm3, %%xmm" << l_n << ", %%xmm" << 4 + l_n << "\\n\\t\"" << std::endl;
    codestream << "                         \"vaddpd %%xmm" << 4 + l_n << ", %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
  }
#if 0
  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%xmm4\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%xmm4\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"movddup " << 8 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movddup (%%r8), %%xmm0\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%xmm0, %%xmm4, %%xmm12\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%xmm12, %%xmm13, %%xmm13\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"movddup " << (8 * call) + (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movddup " << (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%xmm1, %%xmm4, %%xmm11\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%xmm11, %%xmm14, %%xmm14\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"movddup " << (8 * call) + (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movddup " << (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vmulpd %%xmm2, %%xmm4, %%xmm10\\n\\t\"" << std::endl;
  codestream << "                         \"vaddpd %%xmm10, %%xmm15, %%xmm15\\n\\t\"" << std::endl;
#endif
}

void avx1_kernel_1xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast, int max_local_N) {
  codestream << "                         \"vmovsd (%%r9), %%xmm3\\n\\t\"" << std::endl;
 
  for (int l_n = 0; l_n < max_local_N; l_n++) {
    if (l_n == 0) {
      codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
    }
    if (call != -1) {
      codestream << "                         \"vmulsd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%xmm3, %%xmm" << 4 + l_n << "\\n\\t\"" << std::endl;
      codestream << "                         \"vaddsd %%xmm" << 4 + l_n << ", %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmulsd " << (ldb * l_n * 8) << "(%%r8), %%xmm3, %%xmm" << 4 + l_n << "\\n\\t\"" << std::endl;
      codestream << "                         \"vaddsd %%xmm" << 4 + l_n << ", %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
      if (l_n == (max_local_N - 1)) {
        codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
      }
    }
  }

#if 0
  codestream << "                         \"vmovsd (%%r9), %%xmm4\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vmulsd " << 8 * call << "(%%r8), %%xmm4, %%xmm12\\n\\t\"" << std::endl;
    codestream << "                         \"vaddsd %%xmm12, %%xmm13, %%xmm13\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmulsd (%%r8), %%xmm4, %%xmm12\\n\\t\"" << std::endl;
    codestream << "                         \"vaddsd %%xmm12, %%xmm13, %%xmm13\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vmulsd " << (8 * call) + (ldb * 8) << "(%%r8), %%xmm4, %%xmm11\\n\\t\"" << std::endl;
    codestream << "                         \"vaddsd %%xmm11, %%xmm14, %%xmm14\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmulsd " << (ldb * 8) << "(%%r8), %%xmm4, %%xmm11\\n\\t\"" << std::endl;
    codestream << "                         \"vaddsd %%xmm11, %%xmm14, %%xmm14\\n\\t\"" << std::endl;
  }

  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vmulsd " << (8 * call) + (ldb * 16) << "(%%r8), %%xmm4, %%xmm10\\n\\t\"" << std::endl;
    codestream << "                         \"vaddsd %%xmm10, %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmulsd " << (ldb * 16) << "(%%r8), %%xmm4, %%xmm10\\n\\t\"" << std::endl;
    codestream << "                         \"vaddsd %%xmm10, %%xmm15, %%xmm15\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }
#endif
}

void avx1_generate_kernel_dp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch) {
  // functions pointers to different m_blockings
  void (*l_generatorLoad)(std::stringstream&, int, bool, bool, int, std::string);
  void (*l_generatorStore)(std::stringstream&, int, bool, int);
  void (*l_generatorCompute)(std::stringstream&, int, int, int, bool, bool, bool, int, bool, int);

  int k_blocking = 4;
  int k_threshold = 30;
  int mDone, mDone_old;
  int n_blocking = 3;
  int m_blocking = 12;

  init_registers_asm(codestream, tPrefetch);
  header_nloop_dp_asm(codestream, n_blocking);
  
  m_blocking = 12;
  mDone_old = 0;
  mDone = 0;

  // apply m_blocking
  while (mDone != M) {
    mDone_old = mDone;
    mDone = mDone + (((M - mDone_old) / m_blocking) * m_blocking);

    // switch to a different m_blocking
    if (m_blocking == 12) {
      l_generatorLoad = &avx_load_12xN_dp_asm;
      l_generatorStore = &avx_store_12xN_dp_asm;
      l_generatorCompute = &avx1_kernel_12xN_dp_asm;
    } else if (m_blocking == 8) {
      l_generatorLoad = &avx_load_8xN_dp_asm;
      l_generatorStore = &avx_store_8xN_dp_asm;
      l_generatorCompute = &avx1_kernel_8xN_dp_asm;
    } else if (m_blocking == 4) {
      l_generatorLoad = &avx_load_4xN_dp_asm;
      l_generatorStore = &avx_store_4xN_dp_asm;
      l_generatorCompute = &avx1_kernel_4xN_dp_asm;
    } else if (m_blocking == 2) {
      l_generatorLoad = &avx_load_2xN_dp_asm;
      l_generatorStore = &avx_store_2xN_dp_asm;
      l_generatorCompute = &avx1_kernel_2xN_dp_asm;
    } else if (m_blocking == 1) {
      l_generatorLoad = &avx_load_1xN_dp_asm;
      l_generatorStore = &avx_store_1xN_dp_asm;
      l_generatorCompute = &avx1_kernel_1xN_dp_asm;      
    } else {
      std::cout << " !!! ERROR, avx1_generate_kernel_dp, m_blocking is out of range!!! " << std::endl;
      exit(-1);
    }

    if (mDone != mDone_old && mDone > 0) {
      header_mloop_dp_asm(codestream, m_blocking);
      (*l_generatorLoad)(codestream, ldc, alignC, bAdd, n_blocking, tPrefetch);

      if ((K % k_blocking) == 0 && K > k_threshold) {
        header_kloop_dp_asm(codestream, m_blocking, k_blocking);

        for (int k = 0; k < k_blocking; k++) {
	  (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
        }

        footer_kloop_dp_asm(codestream, m_blocking, K);
      } else {
        // we want to fully unroll
        if (K <= k_threshold) {
	  for (int k = 0; k < K; k++) {
	    (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
	  }
        } else {
	  // we want to block, but K % k_blocking != 0
	  int max_blocked_K = (K/k_blocking)*k_blocking;
	  if (max_blocked_K > 0 ) {
	    header_kloop_dp_asm(codestream, m_blocking, k_blocking);
	    for (int k = 0; k < k_blocking; k++) {
	      (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
	    }
	    footer_kloop_notdone_dp_asm(codestream, m_blocking, max_blocked_K );
	  }
	  if (max_blocked_K > 0 ) {
	    codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
	  }
	  for (int k = max_blocked_K; k < K; k++) {
	    (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
	  }
        }
      }

      (*l_generatorStore)(codestream, ldc, alignC, n_blocking);
      footer_mloop_dp_asm(codestream, m_blocking, K, mDone, lda, tPrefetch);
    }

    // switch to a different blocking
    if (m_blocking == 2) {
      m_blocking = 1;
    } else if (m_blocking == 4) {
      m_blocking = 2;
    } else if (m_blocking == 8) {
      m_blocking = 4;
    } else if (m_blocking == 12) {
      m_blocking = 8;
    } else {
      // we are done with m_blocking
    }
  }

#if 0
  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 12);
    avx_load_12xN_dp_asm(codestream, ldc, alignC, bAdd, n_blocking, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 12, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx1_kernel_12xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
      }

      footer_kloop_dp_asm(codestream, 12, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx1_kernel_12xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 12, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx1_kernel_12xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
          }
          footer_kloop_notdone_dp_asm(codestream, 12, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx1_kernel_12xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      }
    }

    avx_store_12xN_dp_asm(codestream, ldc, alignC, n_blocking);
    footer_mloop_dp_asm(codestream, 12, K, mDone, lda, tPrefetch);
  }

  // 8x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 8) * 8);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 8);
    avx_load_8xN_dp_asm(codestream, ldc, alignC, bAdd, n_blocking, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 8, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx1_kernel_8xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
      }

      footer_kloop_dp_asm(codestream, 8, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx1_kernel_8xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 8, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx1_kernel_8xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
          }
          footer_kloop_notdone_dp_asm(codestream, 8, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx1_kernel_8xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      }
    }

    avx_store_8xN_dp_asm(codestream, ldc, alignC, n_blocking);
    footer_mloop_dp_asm(codestream, 8, K, mDone, lda, tPrefetch);
  }

  // 4x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 4) * 4);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 4);
    avx_load_4xN_dp_asm(codestream, ldc, alignC, bAdd, n_blocking, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 4, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx1_kernel_4xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
      }

      footer_kloop_dp_asm(codestream, 4, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx1_kernel_4xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 4, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx1_kernel_4xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
          }
          footer_kloop_notdone_dp_asm(codestream, 4, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx1_kernel_4xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      }
    }

    avx_store_4xN_dp_asm(codestream, ldc, alignC, n_blocking);
    footer_mloop_dp_asm(codestream, 4, K, mDone, lda, tPrefetch);
  }

  // 2x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 2) * 2);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 2);
    avx_load_2xN_dp_asm(codestream, ldc, alignC, bAdd, n_blocking, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 2, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx1_kernel_2xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
      }

      footer_kloop_dp_asm(codestream, 2, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx1_kernel_2xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 2, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx1_kernel_2xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
          }
          footer_kloop_notdone_dp_asm(codestream, 2, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx1_kernel_2xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      }
    }

    avx_store_2xN_dp_asm(codestream, ldc, alignC, n_blocking);
    footer_mloop_dp_asm(codestream, 2, K, mDone, lda, tPrefetch);
  }

  // 1x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 1) * 1);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 1);
    avx_load_1xN_dp_asm(codestream, ldc, alignC, bAdd, n_blocking, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 1, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx1_kernel_1xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
      }

      footer_kloop_dp_asm(codestream, 1, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx1_kernel_1xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 1, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx1_kernel_1xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false, n_blocking);
          }
          footer_kloop_notdone_dp_asm(codestream, 1, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx1_kernel_1xN_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false, n_blocking);
        }
      }
    }

    avx_store_1xN_dp_asm(codestream, ldc, alignC, n_blocking);
    footer_mloop_dp_asm(codestream, 1, K, mDone, lda, tPrefetch);
  }
#endif

  footer_nloop_dp_asm(codestream, n_blocking, N, M, lda, ldb, ldc, tPrefetch);
  close_asm(codestream, tPrefetch);
}

