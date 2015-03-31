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

void sse3_load_6x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, std::string tPrefetch) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"movapd (%%r10), %%xmm7\\n\\t\"" << std::endl;
      codestream << "                         \"movapd 16(%%r10), %%xmm8\\n\\t\"" << std::endl;
      codestream << "                         \"movapd 32(%%r10), %%xmm9\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << ldc * 8 << "(%%r10), %%xmm10\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << (ldc + 2) * 8 << "(%%r10), %%xmm11\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << (ldc + 4) * 8 << "(%%r10), %%xmm12\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << 2 * ldc * 8 << "(%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << ((2 * ldc) + 2) * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << ((2 * ldc) + 4) * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"movupd (%%r10), %%xmm7\\n\\t\"" << std::endl;
      codestream << "                         \"movupd 16(%%r10), %%xmm8\\n\\t\"" << std::endl;
      codestream << "                         \"movupd 32(%%r10), %%xmm9\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << ldc * 8 << "(%%r10), %%xmm10\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << (ldc + 2) * 8 << "(%%r10), %%xmm11\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << (ldc + 4) * 8 << "(%%r10), %%xmm12\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << 2 * ldc * 8 << "(%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << ((2 * ldc) + 2) * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << ((2 * ldc) + 4) * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"xorpd %%xmm7, %%xmm7\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm8, %%xmm8\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm9, %%xmm9\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm10, %%xmm10\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm11, %%xmm11\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm12, %%xmm12\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void sse3_load_4x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"movapd (%%r10), %%xmm10\\n\\t\"" << std::endl;
      codestream << "                         \"movapd 16(%%r10), %%xmm11\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << ldc * 8 << "(%%r10), %%xmm12\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << (ldc + 2) * 8 << "(%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << 2 * ldc * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << ((2 * ldc) + 2) * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"movupd (%%r10), %%xmm10\\n\\t\"" << std::endl;
      codestream << "                         \"movupd 16(%%r10), %%xmm11\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << ldc * 8 << "(%%r10), %%xmm12\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << (ldc + 2) * 8 << "(%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << 2 * ldc * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << ((2 * ldc) + 2) * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"xorpd %%xmm10, %%xmm10\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm11, %%xmm11\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm12, %%xmm12\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void sse3_load_2x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"movapd (%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << ldc * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movapd " << 2 * ldc * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"movupd (%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << ldc * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"movupd " << 2 * ldc * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"xorpd %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"xorpd %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void sse3_load_1x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    codestream << "                         \"movsd (%%r10), %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"movsd " << ldc * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"movsd " << 2 * ldc * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"xorsd %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"xorsd %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"xorsd %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void sse3_store_6x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"movapd %%xmm7, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm8, 16(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm9, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm10, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm11, " << (ldc + 2) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm12, " << (ldc + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm13, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm14, " << ((2 * ldc) + 2) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm15, " << ((2 * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd %%xmm7, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm8, 16(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm9, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm10, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm11, " << (ldc + 2) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm12, " << (ldc + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm13, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm14, " << ((2 * ldc) + 2) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm15, " << ((2 * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void sse3_store_4x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"movapd %%xmm10, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm11, 16(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm12, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm13, " << (ldc + 2) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm14, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm15, " << ((2 * ldc) + 2) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd %%xmm10, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm11, 16(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm12, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm13, " << (ldc + 2) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm14, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm15, " << ((2 * ldc) + 2) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void sse3_store_2x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"movapd %%xmm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm14, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movapd %%xmm15, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd %%xmm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm14, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"movupd %%xmm15, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void sse3_store_1x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  codestream << "                         \"movsd %%xmm13, (%%r10)\\n\\t\"" << std::endl;
  codestream << "                         \"movsd %%xmm14, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
  codestream << "                         \"movsd %%xmm15, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void sse3_kernel_6x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"movddup " << 8 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (8 * call) + (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (8 * call) + (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movddup (%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"movapd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movapd %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm3, %%xmm7\\n\\t\"" << std::endl;
  codestream << "                         \"movapd %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm4, %%xmm10\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm5, %%xmm13\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"movapd 16(%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd 16(%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movapd %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm3, %%xmm8\\n\\t\"" << std::endl;
  codestream << "                         \"movapd %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm4, %%xmm11\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm5, %%xmm14\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"movapd 32(%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd 32(%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movapd %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm3, %%xmm9\\n\\t\"" << std::endl;
  codestream << "                         \"movapd %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm4, %%xmm12\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm5, %%xmm15\\n\\t\"" << std::endl;
}

void sse3_kernel_4x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"movddup " << 8 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (8 * call) + (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (8 * call) + (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movddup (%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"movapd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movapd %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm3, %%xmm10\\n\\t\"" << std::endl;
  codestream << "                         \"movapd %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm4, %%xmm12\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm5, %%xmm14\\n\\t\"" << std::endl;

  if (alignA == true) {
    codestream << "                         \"movapd 16(%%r9), %%xmm6\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd 16(%%r9), %%xmm6\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movapd %%xmm6, %%xmm7\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm0, %%xmm6\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm6, %%xmm11\\n\\t\"" << std::endl;
  codestream << "                         \"movapd %%xmm7, %%xmm8\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm1, %%xmm7\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm7, %%xmm13\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm2, %%xmm8\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm8, %%xmm15\\n\\t\"" << std::endl;
}

void sse3_kernel_2x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"movddup " << 8 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (8 * call) + (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (8 * call) + (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movddup (%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movddup " << (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  if (alignA == true) {
    codestream << "                         \"movapd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movupd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movapd %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm3, %%xmm13\\n\\t\"" << std::endl;
  codestream << "                         \"movapd %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  codestream << "                         \"mulpd %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addpd %%xmm5, %%xmm15\\n\\t\"" << std::endl;
}

void sse3_kernel_1x3_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, int call, bool blast) {
  if (call != -1) {
    codestream << "                         \"movsd " << 8 * call << "(%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movsd " << (8 * call) + (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movsd " << (8 * call) + (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"movsd (%%r8), %%xmm0\\n\\t\"" << std::endl;
    codestream << "                         \"movsd " << (ldb * 8) << "(%%r8), %%xmm1\\n\\t\"" << std::endl;
    codestream << "                         \"movsd " << (ldb * 16) << "(%%r8), %%xmm2\\n\\t\"" << std::endl;
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  codestream << "                         \"movsd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"movapd %%xmm3, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"mulsd %%xmm0, %%xmm3\\n\\t\"" << std::endl;
  codestream << "                         \"addsd %%xmm3, %%xmm13\\n\\t\"" << std::endl;
  codestream << "                         \"movapd %%xmm4, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"mulsd %%xmm1, %%xmm4\\n\\t\"" << std::endl;
  codestream << "                         \"addsd %%xmm4, %%xmm14\\n\\t\"" << std::endl;
  codestream << "                         \"mulsd %%xmm2, %%xmm5\\n\\t\"" << std::endl;
  codestream << "                         \"addsd %%xmm5, %%xmm15\\n\\t\"" << std::endl;
}

void sse3_generate_kernel_dp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch) {
  int k_blocking = 100;
  int k_threshold = 30;
  int mDone, mDone_old;
  init_registers_asm(codestream, tPrefetch);
  header_nloop_dp_asm(codestream, 3);
  // 6x3
  mDone_old = 0;
  mDone = (M / 6) * 6;

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 6);
    sse3_load_6x3_dp_asm(codestream, ldc, alignC, bAdd, tPrefetch);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 6, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        sse3_kernel_6x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 6, K);
    } else {
      for (int k = 0; k < K; k++) {
        sse3_kernel_6x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
      }
    }

    sse3_store_6x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 6, K, mDone, lda, tPrefetch);
  }

  // 4x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 4) * 4);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 4);
    sse3_load_4x3_dp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 4, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        sse3_kernel_4x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 4, K);
    } else {
      for (int k = 0; k < K; k++) {
        sse3_kernel_4x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
      }
    }

    sse3_store_4x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 4, K, mDone, lda, tPrefetch);
  }

  // 2x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 2) * 2);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 2);
    sse3_load_2x3_dp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 2, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        sse3_kernel_2x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 2, K);
    } else {
      for (int k = 0; k < K; k++) {
        sse3_kernel_2x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
      }
    }

    sse3_store_2x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 2, K, mDone, lda, tPrefetch);
  }

  // 1x3
  mDone_old = mDone;
  mDone = mDone + (((M - mDone_old) / 1) * 1);

  if (mDone != mDone_old && mDone > 0) {
    header_mloop_dp_asm(codestream, 1);
    sse3_load_1x3_dp_asm(codestream, ldc, alignC, bAdd);

    if ((K % k_blocking) == 0 && K > k_threshold) {
      header_kloop_dp_asm(codestream, 1, k_blocking);

      for (int k = 0; k < k_blocking; k++) {
        sse3_kernel_1x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, -1, false);
      }

      footer_kloop_dp_asm(codestream, 1, K);
    } else {
      for (int k = 0; k < K; k++) {
        sse3_kernel_1x3_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, k, false);
      }
    }

    sse3_store_1x3_dp_asm(codestream, ldc, alignC, tPrefetch);
    footer_mloop_dp_asm(codestream, 1, K, mDone, lda, tPrefetch);
  }

  footer_nloop_dp_asm(codestream, 3, N, M, lda, ldb, ldc, tPrefetch);
  close_asm(codestream, tPrefetch);
}

