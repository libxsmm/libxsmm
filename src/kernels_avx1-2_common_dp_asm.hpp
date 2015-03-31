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

void avx_load_12x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, std::string tPrefetch) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovapd (%%r10), %%ymm7\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd 32(%%r10), %%ymm8\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd 64(%%r10), %%ymm9\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ldc * 8 << "(%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << (ldc + 4) * 8 << "(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << (ldc + 8) * 8 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << 2 * ldc * 8 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ((2 * ldc) + 4) * 8 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ((2 * ldc) + 8) * 8 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovupd (%%r10), %%ymm7\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd 32(%%r10), %%ymm8\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd 64(%%r10), %%ymm9\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ldc * 8 << "(%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << (ldc + 4) * 8 << "(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << (ldc + 8) * 8 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << 2 * ldc * 8 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ((2 * ldc) + 4) * 8 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ((2 * ldc) + 8) * 8 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    }
  } else {
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

void avx_load_8x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, std::string tPrefetch) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovapd (%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd 32(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ldc * 8 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << (ldc + 4) * 8 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << 2 * ldc * 8 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ((2 * ldc) + 4) * 8 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovupd (%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd 32(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ldc * 8 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << (ldc + 4) * 8 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << 2 * ldc * 8 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ((2 * ldc) + 4) * 8 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"vxorpd %%ymm10, %%ymm10, %%ymm10\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm11, %%ymm11, %%ymm11\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm12, %%ymm12, %%ymm12\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm13, %%ymm13, %%ymm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm14, %%ymm14, %%ymm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm15, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_load_4x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovapd (%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ldc * 8 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << 2 * ldc * 8 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovupd (%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ldc * 8 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << 2 * ldc * 8 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"vxorpd %%ymm13, %%ymm13, %%ymm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm14, %%ymm14, %%ymm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%ymm15, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
  }
}

void avx_load_2x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovapd (%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << ldc * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd " << 2 * ldc * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovupd (%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << ldc * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd " << 2 * ldc * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"vxorpd %%xmm13, %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%xmm14, %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%xmm15, %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void avx_load_1x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    codestream << "                         \"vmovsd (%%r10), %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"vmovsd " << ldc * 8 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"vmovsd " << 2 * ldc * 8 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vxorpd %%xmm13, %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%xmm14, %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorpd %%xmm15, %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void avx_store_12x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovapd %%ymm7, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm8, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm9, 64(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm10, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm11, " << (ldc + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm12, " << (ldc + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm13, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm14, " << ((2 * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm15, " << ((2 * ldc) + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd %%ymm7, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm8, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm9, 64(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm10, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm11, " << (ldc + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm12, " << (ldc + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm13, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm14, " << ((2 * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm15, " << ((2 * ldc) + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 " << (ldc + 8) * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ((2 * ldc) + 8) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_store_8x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovapd %%ymm10, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm11, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm12, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm13, " << (ldc + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm14, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm15, " << ((2 * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd %%ymm10, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm11, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm12, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm13, " << (ldc + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm14, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm15, " << ((2 * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_store_4x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovapd %%ymm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm14, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%ymm15, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd %%ymm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm14, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%ymm15, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_store_2x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovapd %%xmm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%xmm14, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovapd %%xmm15, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd %%xmm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%xmm14, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovupd %%xmm15, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_store_1x3_dp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  codestream << "                         \"vmovsd %%xmm13, (%%r10)\\n\\t\"" << std::endl;
  codestream << "                         \"vmovsd %%xmm14, " << ldc * 8 << "(%%r10)\\n\\t\"" << std::endl;
  codestream << "                         \"vmovsd %%xmm15, " << (2 * ldc) * 8 << "(%%r10)\\n\\t\"" << std::endl;
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 8 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 8 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

