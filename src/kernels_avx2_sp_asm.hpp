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

void avx2_load_32x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, std::string tPrefetch) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovaps (%%r10), %%ymm4\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps 32(%%r10), %%ymm5\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps 64(%%r10), %%ymm6\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps 96(%%r10), %%ymm7\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ldc * 4 << "(%%r10), %%ymm8\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << (ldc + 8) * 4 << "(%%r10), %%ymm9\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << (ldc + 16) * 4 << "(%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << (ldc + 24) * 4 << "(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << 2 * ldc * 4 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ((2 * ldc) + 8) * 4 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ((2 * ldc) + 16) * 4 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ((2 * ldc) + 24) * 4 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovups (%%r10), %%ymm4\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups 32(%%r10), %%ymm5\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups 64(%%r10), %%ymm6\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups 96(%%r10), %%ymm7\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ldc * 4 << "(%%r10), %%ymm8\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << (ldc + 8) * 4 << "(%%r10), %%ymm9\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << (ldc + 16) * 4 << "(%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << (ldc + 24) * 4 << "(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << 2 * ldc * 4 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ((2 * ldc) + 8) * 4 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ((2 * ldc) + 16) * 4 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ((2 * ldc) + 24) * 4 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"vxorps %%ymm4, %%ymm4, %%ymm4\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm5, %%ymm5, %%ymm5\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm6, %%ymm6, %%ymm6\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm7, %%ymm7, %%ymm7\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm8, %%ymm8, %%ymm8\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm9, %%ymm9, %%ymm9\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm10, %%ymm10, %%ymm10\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm11, %%ymm11, %%ymm11\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm12, %%ymm12, %%ymm12\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm13, %%ymm13, %%ymm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm14, %%ymm14, %%ymm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm15, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 64(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx2_store_32x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovaps %%ymm4, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm5, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm6, 64(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm7, 96(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm8, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm9, " << (ldc + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm10, " << (ldc + 16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm11, " << (ldc + 24) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm12, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm13, " << ((2 * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm14, " << ((2 * ldc) + 16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm15, " << ((2 * ldc) + 24) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups %%ymm4, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm5, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm6, 64(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm7, 96(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm8, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm9, " << (ldc + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm10, " << (ldc + 16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm11, " << (ldc + 24) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm12, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm13, " << ((2 * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm14, " << ((2 * ldc) + 16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm15, " << ((2 * ldc) + 24) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 " << (ldc + 16) * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ((2 * ldc) + 16) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx2_kernel_32x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != (-1)) {
    codestream << "                         \"vbroadcastss " << 4 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 4) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 8) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss (%%r8), %%ymm0\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastss " << (ldb * 4) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastss " << (ldb * 8) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  // first row
  if (alignA == true) {
    codestream << "                         \"vmovaps (%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups (%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm0, %%ymm4\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm1, %%ymm8\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm2, %%ymm12\\n\\t\"" << std::endl;

  // second row
  if (alignA == true) {
    codestream << "                         \"vmovaps 32(%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups 32(%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm0, %%ymm5\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm1, %%ymm9\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm2, %%ymm13\\n\\t\"" << std::endl;

  //third row
  if (alignA == true) {
    codestream << "                         \"vmovaps 64(%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups 64(%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm0, %%ymm6\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm1, %%ymm10\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm2, %%ymm14\\n\\t\"" << std::endl;

  //fourth row
  if (alignA == true) {
    codestream << "                         \"vmovaps 96(%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups 96(%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"addq $" << lda * 4 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm0, %%ymm7\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm1, %%ymm11\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm2, %%ymm15\\n\\t\"" << std::endl;
}

void avx2_kernel_24x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"vbroadcastss " << 4 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss (%%r8), %%ymm0\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"vmovaps (%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups (%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 4) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 8) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss " << (ldb * 4) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
    codestream << "                         \"vbroadcastss " << (ldb * 8) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm0, %%ymm7\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm1, %%ymm10\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"vmovaps 32(%%r9), %%ymm4\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups 32(%%r9), %%ymm4\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm3, %%ymm2, %%ymm13\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm4, %%ymm0, %%ymm8\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm4, %%ymm1, %%ymm11\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"vmovaps 64(%%r9), %%ymm5\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups 64(%%r9), %%ymm5\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm4, %%ymm2, %%ymm14\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm5, %%ymm0, %%ymm9\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm5, %%ymm1, %%ymm12\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm5, %%ymm2, %%ymm15\\n\\t\"" << std::endl;
}

void avx2_kernel_16x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (alignA == true) {
    codestream << "                         \"vmovaps (%%r9), %%ymm4\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps 32(%%r9), %%ymm5\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups (%%r9), %%ymm4\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups 32(%%r9), %%ymm5\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << 4 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss (%%r8), %%ymm0\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm0, %%ymm4, %%ymm10\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm0, %%ymm5, %%ymm11\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 4) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss " << (ldb * 4) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm1, %%ymm4, %%ymm12\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm1, %%ymm5, %%ymm13\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 8) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss " << (ldb * 8) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm2, %%ymm4, %%ymm14\\n\\t\"" << std::endl;
  codestream << "                         \"vfmadd231ps %%ymm2, %%ymm5, %%ymm15\\n\\t\"" << std::endl;
}

void avx2_kernel_8x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (alignA == true) {
    codestream << "                         \"vmovaps (%%r9), %%ymm4\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups (%%r9), %%ymm4\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << 4 * call << "(%%r8), %%ymm0\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss (%%r8), %%ymm0\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm0, %%ymm4, %%ymm13\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 4) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss " << (ldb * 4) << "(%%r8), %%ymm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm1, %%ymm4, %%ymm14\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 8) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss " << (ldb * 8) << "(%%r8), %%ymm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%ymm2, %%ymm4, %%ymm15\\n\\t\"" << std::endl;
}

void avx2_kernel_4x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (alignA == true) {
    codestream << "                         \"vmovaps (%%r9), %%xmm4\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups (%%r9), %%xmm4\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << 4 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss (%%r8), %%xmm0\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%xmm0, %%xmm4, %%xmm13\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss " << (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%xmm1, %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vbroadcastss " << (4 * call) + (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vbroadcastss " << (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"vfmadd231ps %%xmm2, %%xmm4, %%xmm15\\n\\t\"" << std::endl;
}

void avx2_kernel_1x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  codestream << "                         \"vmovss (%%r9), %%xmm4\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vfmadd231ss " << 4 * call << "(%%r8), %%xmm4, %%xmm13\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vfmadd231ss (%%r8), %%xmm4, %%xmm13\\n\\t\"" << std::endl;
  }

  if (call != -1) {
    codestream << "                         \"vfmadd231ss " << (4 * call) + (ldb * 4) << "(%%r8), %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vfmadd231ss " << (ldb * 4) << "(%%r8), %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  }

  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;

  if (call != -1) {
    codestream << "                         \"vfmadd231ss " << (4 * call) + (ldb * 8) << "(%%r8), %%xmm4, %%xmm15\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vfmadd231ss " << (ldb * 8) << "(%%r8), %%xmm4, %%xmm15\\n\\t\"" << std::endl;
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }
}

void avx2_generate_kernel_sp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch) {
  int k_blocking = 4;
  int k_threshold = 30;
  int mDone, mDone_old;
  init_registers_asm(codestream, tPrefetch);
  header_nloop_sp_asm(codestream, 3);

  // 32x3
  mDone_old = 0;
  mDone = (M / 32) * 32;

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 32);
    avx2_load_32x3_sp_asm(codestream, ldc, alignC, bAdd, tPrefetch);

    if (K % k_blocking == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 32, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_32x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 32, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_32x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_sp_asm(codestream, 32, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_32x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_sp_asm(codestream, 32, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 4 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_32x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx2_store_32x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 32, K, mDone, lda, tPrefetch);
  }

  // 24x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 24) * 24);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 24);
    avx_load_24x3_sp_asm(codestream, ldc, alignC, bAdd, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 24, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_24x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 24, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_24x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_sp_asm(codestream, 24, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_24x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_sp_asm(codestream, 24, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 4 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_24x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_24x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 24, K, mDone, lda, tPrefetch);
  }

  // 16x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 16) * 16);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 16);
    avx_load_16x3_sp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 16, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_16x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 16, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_16x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_sp_asm(codestream, 16, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_16x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_sp_asm(codestream, 16, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 4 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_16x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_16x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 16, K, mDone, lda, tPrefetch);
  }

  // 8x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 8) * 8);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 8);
    avx_load_8x3_sp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 8, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_8x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 8, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_8x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_sp_asm(codestream, 8, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_8x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_sp_asm(codestream, 8, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 4 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_8x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_8x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 8, K, mDone, lda, tPrefetch);
  }

  // 4x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 4) * 4);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 4);
    avx_load_4x3_sp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 4, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_4x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 4, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_4x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_sp_asm(codestream, 4, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_4x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_sp_asm(codestream, 4, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 4 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_4x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_4x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 4, K, mDone, lda, tPrefetch);
  }

  // 1x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 1) * 1);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 1);
    avx_load_1x3_sp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 1, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        avx2_kernel_1x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 1, K);
    } else {
      // we want to fully unroll
      if (K <= k_threshold) {
        for (int k = 0; k < K; k++) {
          avx2_kernel_1x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      } else {
        // we want to block, but K % k_blocking != 0
        int max_blocked_K = (K/k_blocking)*k_blocking;
        if (max_blocked_K > 0 ) {
          header_kloop_sp_asm(codestream, 1, k_blocking);
          for (int k = 0; k < k_blocking; k++) {
            avx2_kernel_1x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
          }
          footer_kloop_notdone_sp_asm(codestream, 1, max_blocked_K );
        }
        if (max_blocked_K > 0 ) {
          codestream << "                         \"subq $" << max_blocked_K * 4 << ", %%r8\\n\\t\"" << std::endl;
        }
        for (int k = max_blocked_K; k < K; k++) {
          avx2_kernel_1x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
        }
      }
    }

    avx_store_1x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 1, K, mDone, lda, tPrefetch);
  }

  footer_nloop_sp_asm(codestream, 3, N, M, lda, ldb, ldc, tPrefetch);
  close_asm(codestream, tPrefetch);
}

