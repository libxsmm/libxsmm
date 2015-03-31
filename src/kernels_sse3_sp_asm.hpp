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

void sse3_load_12x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, std::string tPrefetch) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"movaps (%%r10), %%xmm7\\n\\t\"" << std::endl;
      codestream << "                         \"movaps 16(%%r10), %%xmm8\\n\\t\"" << std::endl;
      codestream << "                         \"movaps 32(%%r10), %%xmm9\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << ldc * 4 << "(%%r10), %%xmm10\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << (ldc + 4) * 4 << "(%%r10), %%xmm11\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << (ldc + 8) * 4 << "(%%r10), %%xmm12\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << 2 * ldc * 4 << "(%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << ((2 * ldc) + 4) * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << ((2 * ldc) + 8) * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"movups (%%r10), %%xmm7\\n\\t\"" << std::endl;
      codestream << "                         \"movups 16(%%r10), %%xmm8\\n\\t\"" << std::endl;
      codestream << "                         \"movups 32(%%r10), %%xmm9\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << ldc * 4 << "(%%r10), %%xmm10\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << (ldc + 4) * 4 << "(%%r10), %%xmm11\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << (ldc + 8) * 4 << "(%%r10), %%xmm12\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << 2 * ldc * 4 << "(%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << ((2 * ldc) + 4) * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << ((2 * ldc) + 8) * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"xorps %%xmm7, %%xmm7\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm8, %%xmm8\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm9, %%xmm9\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm10, %%xmm10\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm11, %%xmm11\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm12, %%xmm12\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void sse3_load_8x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"movaps (%%r10), %%xmm10\\n\\t\"" << std::endl;
      codestream << "                         \"movaps 16(%%r10), %%xmm11\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << ldc * 4 << "(%%r10), %%xmm12\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << (ldc + 4) * 4 << "(%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << 2 * ldc * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << ((2 * ldc) + 4) * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"movups (%%r10), %%xmm10\\n\\t\"" << std::endl;
      codestream << "                         \"movups 16(%%r10), %%xmm11\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << ldc * 4 << "(%%r10), %%xmm12\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << (ldc + 4) * 4 << "(%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << 2 * ldc * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << ((2 * ldc) + 4) * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"xorps %%xmm10, %%xmm10\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm11, %%xmm11\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm12, %%xmm12\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void sse3_load_4x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"movaps (%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << ldc * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movaps " << 2 * ldc * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"movups (%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << ldc * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movups " << 2 * ldc * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"xorps %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void sse3_load_1x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    codestream << "                         \"movss (%%r10), %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"movss " << ldc * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"movss " << 2 * ldc * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"xorps %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"xorps %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void sse3_store_12x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"movaps %%xmm7, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm8, 16(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm9, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm10, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm11, " << (ldc + 4) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm12, " << (ldc + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm13, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm14, " << ((2 * ldc) + 4) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm15, " << ((2 * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups %%xmm7, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm8, 16(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm9, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm10, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm11, " << (ldc + 4) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm12, " << (ldc + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm13, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm14, " << ((2 * ldc) + 4) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm15, " << ((2 * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void sse3_store_8x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"movaps %%xmm10, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm11, 16(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm12, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm13, " << (ldc + 4) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm14, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm15, " << ((2 * ldc) + 4) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups %%xmm10, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm11, 16(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm12, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm13, " << (ldc + 4) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm14, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm15, " << ((2 * ldc) + 4) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void sse3_store_4x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"movaps %%xmm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm14, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movaps %%xmm15, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups %%xmm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm14, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movups %%xmm15, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void sse3_store_1x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  codestream << "                         \"movss %%xmm13, (%%r10)\\n\\t\"" << std::endl;
  codestream << "                         \"movss %%xmm14, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
  codestream << "                         \"movss %%xmm15, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void sse3_kernel_12x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"movss " << 4 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm0, %%xmm0\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (4 * call) + (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm1, %%xmm1\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (4 * call) + (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm2, %%xmm2\\n\\t\"" << std::endl; 
  } else {
    codestream << "                         \"movss (%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm0, %%xmm0\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm1, %%xmm1\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm2, %%xmm2\\n\\t\"" << std::endl; 
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"movaps (%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups (%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movaps %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm3, %%xmm7\\n\\t\"" << std::endl;
  codestream << "                         \"movaps %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm4, %%xmm10\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm5, %%xmm13\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"movaps 16(%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups 16(%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movaps %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm3, %%xmm8\\n\\t\"" << std::endl;
  codestream << "                         \"movaps %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm4, %%xmm11\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm5, %%xmm14\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"movaps 32(%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups 32(%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movaps %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm3, %%xmm9\\n\\t\"" << std::endl;
  codestream << "                         \"movaps %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm4, %%xmm12\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm5, %%xmm15\\n\\t\"" << std::endl;
}

void sse3_kernel_8x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"movss " << 4 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm0, %%xmm0\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (4 * call) + (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm1, %%xmm1\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (4 * call) + (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm2, %%xmm2\\n\\t\"" << std::endl; 
  } else {
    codestream << "                         \"movss (%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm0, %%xmm0\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm1, %%xmm1\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm2, %%xmm2\\n\\t\"" << std::endl; 
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"movaps (%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups (%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movaps %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm3, %%xmm10\\n\\t\"" << std::endl;
  codestream << "                         \"movaps %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm4, %%xmm12\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm5, %%xmm14\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"movaps 16(%%r9), %%xmm6\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups 16(%%r9), %%xmm6\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movaps %%xmm6, %%xmm7\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm0, %%xmm6\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm6, %%xmm11\\n\\t\"" << std::endl;
  codestream << "                         \"movaps %%xmm7, %%xmm8\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm1, %%xmm7\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm7, %%xmm13\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm2, %%xmm8\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm8, %%xmm15\\n\\t\"" << std::endl;
}

void sse3_kernel_4x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"movss " << 4 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm0, %%xmm0\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (4 * call) + (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm1, %%xmm1\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (4 * call) + (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm2, %%xmm2\\n\\t\"" << std::endl; 
  } else {
    codestream << "                         \"movss (%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm0, %%xmm0\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm1, %%xmm1\\n\\t\"" << std::endl; 
    codestream << "                         \"movss " << (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"shufps $0x00, %%xmm2, %%xmm2\\n\\t\"" << std::endl; 
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"movaps (%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movups (%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movaps %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm3, %%xmm13\\n\\t\"" << std::endl;
  codestream << "                         \"movaps %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  codestream << "                         \"mulps %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addps %%xmm5, %%xmm15\\n\\t\"" << std::endl;
}

void sse3_kernel_1x3_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"movss " << 4 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movss " << (4 * call) + (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movss " << (4 * call) + (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movss (%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movss " << (ldb * 4) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movss " << (ldb * 8) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movss (%%r9), %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"movaps %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulss %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addss %%xmm3, %%xmm13\\n\\t\"" << std::endl;
  codestream << "                         \"movaps %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 4 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"mulss %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addss %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  codestream << "                         \"mulss %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addss %%xmm5, %%xmm15\\n\\t\"" << std::endl;
}

void sse3_generate_kernel_sp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch) {
  int k_blocking = 100;
  int k_threshold = 30;
  int mDone, mDone_old;
  init_registers_asm(codestream, tPrefetch);
  header_nloop_sp_asm(codestream, 3);
  // 12x3
  mDone_old = 0;
  mDone = (M / 12) * 12;

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 12);
    sse3_load_12x3_sp_asm(codestream, ldc, alignC, bAdd, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 12, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        sse3_kernel_12x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 12, K);
    } else {
      for (int k = 0; k < K; k++) {
        sse3_kernel_12x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
      }
    }

    sse3_store_12x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 12, K, mDone, lda, tPrefetch);
  }

  // 8x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 8) * 8);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 8);
    sse3_load_8x3_sp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 8, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        sse3_kernel_8x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 8, K);
    } else {
      for (int k = 0; k < K; k++) {
        sse3_kernel_8x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
      }
    }

    sse3_store_8x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 8, K, mDone, lda, tPrefetch);
  }

  // 4x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 4) * 4);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 4);
    sse3_load_4x3_sp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 4, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        sse3_kernel_4x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 4, K);
    } else {
      for (int k = 0; k < K; k++) {
        sse3_kernel_4x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
      }
    }

    sse3_store_4x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 4, K, mDone, lda, tPrefetch);
  }

  // 1x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 1) * 1);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_sp_asm(codestream, 1);
    sse3_load_1x3_sp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_sp_asm(codestream, 1, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        sse3_kernel_1x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_sp_asm(codestream, 1, K);
    } else {
      for (int k = 0; k < K; k++) {
        sse3_kernel_1x3_sp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
      }
    }

    sse3_store_1x3_sp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_sp_asm(codestream, 1, K, mDone, lda, tPrefetch);
  }

  footer_nloop_sp_asm(codestream, 3, N, M, lda, ldb, ldc, tPrefetch);
  close_asm(codestream, tPrefetch);
}

