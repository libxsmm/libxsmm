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

void avx2_load_16x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, std::string tPrefetch) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovapd (%%r10), %%ymm4\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd 32(%%r10), %%ymm5\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd 64(%%r10), %%ymm6\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd 96(%%r10), %%ymm7\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ldc * 8 << "(%%r10), %%ymm8\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << (ldc + 4) * 8 << "(%%r10), %%ymm9\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << (ldc + 8) * 8 << "(%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << (ldc + 12) * 8 << "(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << 2 * ldc * 8 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ((2 * ldc) + 4) * 8 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ((2 * ldc) + 8) * 8 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ((2 * ldc) + 12) * 8 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovupd (%%r10), %%ymm4\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd 32(%%r10), %%ymm5\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd 64(%%r10), %%ymm6\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd 96(%%r10), %%ymm7\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ldc * 8 << "(%%r10), %%ymm8\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << (ldc + 4) * 8 << "(%%r10), %%ymm9\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << (ldc + 8) * 8 << "(%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << (ldc + 12) * 8 << "(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << 2 * ldc * 8 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ((2 * ldc) + 4) * 8 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ((2 * ldc) + 8) * 8 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ((2 * ldc) + 12) * 8 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"vxorpd %%ymm4, %%ymm4, %%ymm4\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm5, %%ymm5, %%ymm5\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm6, %%ymm6, %%ymm6\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm7, %%ymm7, %%ymm7\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm8, %%ymm8, %%ymm8\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm9, %%ymm9, %%ymm9\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm10, %%ymm10, %%ymm10\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm11, %%ymm11, %%ymm11\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm12, %%ymm12, %%ymm12\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm13, %%ymm13, %%ymm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm14, %%ymm14, %%ymm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm15, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 64(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx2_store_16x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovapd %%ymm4, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm5, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm6, 64(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm7, 96(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm8, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm9, " << (ldc + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm10, " << (ldc + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm11, " << (ldc + 12) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm12, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm13, " << ((2 * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm14, " << ((2 * ldc) + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm15, " << ((2 * ldc) + 12) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd %%ymm4, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm5, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm6, 64(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm7, 96(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm8, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm9, " << (ldc + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm10, " << (ldc + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm11, " << (ldc + 12) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm12, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm13, " << ((2 * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm14, " << ((2 * ldc) + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm15, " << ((2 * ldc) + 12) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 " << (ldc + 8) * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ((2 * ldc) + 8) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx2_kernel_16x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != (-1)) {
    codestream << "                         \"vbroadcastsd " << 8 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd (%%r8), %%ymm0\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastsd " << (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastsd " << (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  // first row
  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm0, %%ymm4\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm1, %%ymm8\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm2, %%ymm12\\n\\t\"" << std::endl;

  // second row
  if (alignA == true) {
    codestream << "                         \"vmovapd 32(%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd 32(%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm0, %%ymm5\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm1, %%ymm9\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm2, %%ymm13\\n\\t\"" << std::endl;

  //third row
  if (alignA == true) {
    codestream << "                         \"vmovapd 64(%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd 64(%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm0, %%ymm6\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm1, %%ymm10\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm2, %%ymm14\\n\\t\"" << std::endl;

  //fourth row
  if (alignA == true) {
    codestream << "                         \"vmovapd 96(%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd 96(%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"addq $" << lda * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm0, %%ymm7\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm1, %%ymm11\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm2, %%ymm15\\n\\t\"" << std::endl;
}

void avx2_kernel_12x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << 8 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd (%%r8), %%ymm0\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastsd " << (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm0, %%ymm7\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm1, %%ymm10\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"vmovapd 32(%%r9), %%ymm4\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd 32(%%r9), %%ymm4\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm3, %%ymm2, %%ymm13\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm4, %%ymm0, %%ymm8\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm4, %%ymm1, %%ymm11\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"vmovapd 64(%%r9), %%ymm5\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd 64(%%r9), %%ymm5\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm4, %%ymm2, %%ymm14\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm5, %%ymm0, %%ymm9\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm5, %%ymm1, %%ymm12\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm5, %%ymm2, %%ymm15\\n\\t\"" << std::endl;
}

void avx2_kernel_8x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
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

  codestream << "                         \"vfmadd231pd %%ymm0, %%ymm4, %%ymm10\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm0, %%ymm5, %%ymm11\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm1, %%ymm4, %%ymm12\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm1, %%ymm5, %%ymm13\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm2, %%ymm4, %%ymm14\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231pd %%ymm2, %%ymm5, %%ymm15\\n\\t\"" << std::endl;
}

void avx2_kernel_4x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
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

  codestream << "                         \"vfmadd231pd %%ymm0, %%ymm4, %%ymm13\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 8) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm1, %%ymm4, %%ymm14\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastsd " << (ldb * 16) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%ymm2, %%ymm4, %%ymm15\\n\\t\"" << std::endl;
}

void avx2_kernel_2x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
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

  codestream << "                         \"vfmadd231pd %%xmm0, %%xmm4, %%xmm13\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"movddup " << (8 * call) + (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movddup " << (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%xmm1, %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"movddup " << (8 * call) + (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movddup " << (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231pd %%xmm2, %%xmm4, %%xmm15\\n\\t\"" << std::endl;
}

void avx2_kernel_1x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  codestream << "                         \"vmovsd (%%r9), %%xmm4\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vfmadd231sd " << 8 * call << "(%%r8), %%xmm4, %%xmm13\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vfmadd231sd (%%r8), %%xmm4, %%xmm13\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vfmadd231sd " << (8 * call) + (ldb * 8) << "(%%r8), %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vfmadd231sd " << (ldb * 8) << "(%%r8), %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  }

  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vfmadd231sd " << (8 * call) + (ldb * 16) << "(%%r8), %%xmm4, %%xmm15\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vfmadd231sd " << (ldb * 16) << "(%%r8), %%xmm4, %%xmm15\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }
}

void avx2_generate_kernel_dp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch) {
  int k_blocking = 4;
  int k_threshold = 30;
  int mDone, mDone_old;
  init_registers_asm(codestream, tPrefetch);
  header_nloop_dp_asm(codestream, 3);

  mDone_old = 0;
  // 16x3
  if (M == 56) {
    mDone = 32;
  } else {
    mDone = (M / 16) * 16;
  }

  if (mDone > 0) {
    header_mloop_dp_asm(codestream, 16);
    avx2_load_16x3_dp_asm(codestream, ldc, alignC, bAdd, tPrefetch);

    if (K % k_blocking == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 16, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_16x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 16, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_16x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 16, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_16x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_dp_asm(codestream, 16, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_16x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx2_store_16x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 16, K, mDone, lda, tPrefetch);
  }

  // 12x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 12) * 12);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 12);
    avx_load_12x3_dp_asm(codestream, ldc, alignC, bAdd, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 12, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_12x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 12, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_12x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 12, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_12x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_dp_asm(codestream, 12, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_12x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_12x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 12, K, mDone, lda, tPrefetch);
  }

  // 8x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 8) * 8);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 8);
    avx_load_8x3_dp_asm(codestream, ldc, alignC, bAdd, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 8, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_8x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 8, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_8x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 8, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_8x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_dp_asm(codestream, 8, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_8x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_8x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 8, K, mDone, lda, tPrefetch);
  }

  // 4x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 4) * 4);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 4);
    avx_load_4x3_dp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 4, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_4x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 4, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_4x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 4, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_4x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_dp_asm(codestream, 4, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_4x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_4x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 4, K, mDone, lda, tPrefetch);
  }

  // 2x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 2) * 2);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 2);
    avx_load_2x3_dp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 2, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_2x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 2, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_2x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 2, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_2x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_dp_asm(codestream, 2, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_2x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_2x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 2, K, mDone, lda, tPrefetch);
  }

  // 1x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 1) * 1);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 1);
    avx_load_1x3_dp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 1, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_1x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 1, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_1x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_dp_asm(codestream, 1, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_1x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_dp_asm(codestream, 1, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_1x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_1x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 1, K, mDone, lda, tPrefetch);
  }

  footer_nloop_dp_asm(codestream, 3, N, M, lda, ldb, ldc, tPrefetch);
  close_asm(codestream, tPrefetch);
}

