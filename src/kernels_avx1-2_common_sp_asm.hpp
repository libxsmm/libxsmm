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

void avx_load_24x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, std::string tPrefetch) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovaps (%%r10), %%ymm7\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps 32(%%r10), %%ymm8\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps 64(%%r10), %%ymm9\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ldc * 4 << "(%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << (ldc + 8) * 4 << "(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << (ldc + 16) * 4 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << 2 * ldc * 4 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ((2 * ldc) + 8) * 4 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ((2 * ldc) + 16) * 4 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovups (%%r10), %%ymm7\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups 32(%%r10), %%ymm8\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups 64(%%r10), %%ymm9\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ldc * 4 << "(%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << (ldc + 8) * 4 << "(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << (ldc + 16) * 4 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << 2 * ldc * 4 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ((2 * ldc) + 8) * 4 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ((2 * ldc) + 16) * 4 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    }
  } else {
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
}

void avx_load_16x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovaps (%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps 32(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ldc * 4 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << (ldc + 8) * 4 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << 2 * ldc * 4 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ((2 * ldc) + 8) * 4 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovups (%%r10), %%ymm10\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups 32(%%r10), %%ymm11\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ldc * 4 << "(%%r10), %%ymm12\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << (ldc + 8) * 4 << "(%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << 2 * ldc * 4 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ((2 * ldc) + 8) * 4 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"vxorps %%ymm10, %%ymm10, %%ymm10\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm11, %%ymm11, %%ymm11\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm12, %%ymm12, %%ymm12\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm13, %%ymm13, %%ymm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm14, %%ymm14, %%ymm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm15, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
  }
}

void avx_load_8x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovaps (%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ldc * 4 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << 2 * ldc * 4 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovups (%%r10), %%ymm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ldc * 4 << "(%%r10), %%ymm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << 2 * ldc * 4 << "(%%r10), %%ymm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"vxorps %%ymm13, %%ymm13, %%ymm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm14, %%ymm14, %%ymm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%ymm15, %%ymm15, %%ymm15\\n\\t\"" << std::endl;
  }
}

void avx_load_4x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    if (alignC == true) {
      codestream << "                         \"vmovaps (%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << ldc * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps " << 2 * ldc * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovups (%%r10), %%xmm13\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << ldc * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups " << 2 * ldc * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
    }
  } else {
    codestream << "                         \"vxorps %%xmm13, %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%xmm14, %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%xmm15, %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void avx_load_1x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd) {
  if (bAdd) {
    codestream << "                         \"vmovss (%%r10), %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"vmovss " << ldc * 4 << "(%%r10), %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"vmovss " << 2 * ldc * 4 << "(%%r10), %%xmm15\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vxorps %%xmm13, %%xmm13, %%xmm13\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%xmm14, %%xmm14, %%xmm14\\n\\t\"" << std::endl;
    codestream << "                         \"vxorps %%xmm15, %%xmm15, %%xmm15\\n\\t\"" << std::endl;
  }
}

void avx_store_24x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovaps %%ymm7, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm8, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm9, 64(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm10, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm11, " << (ldc + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm12, " << (ldc + 16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm13, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm14, " << ((2 * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm15, " << ((2 * ldc) + 16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups %%ymm7, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm8, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm9, 64(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm10, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm11, " << (ldc + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm12, " << (ldc + 16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm13, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm14, " << ((2 * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm15, " << ((2 * ldc) + 16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 64(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (ldc + 16) * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ((2 * ldc) + 16) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_store_16x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovaps %%ymm10, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm11, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm12, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm13, " << (ldc + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm14, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm15, " << ((2 * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups %%ymm10, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm11, 32(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm12, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm13, " << (ldc + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm14, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm15, " << ((2 * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_store_8x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovaps %%ymm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm14, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%ymm15, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups %%ymm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm14, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%ymm15, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_store_4x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  if (alignC == true) {
    codestream << "                         \"vmovaps %%xmm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%xmm14, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovaps %%xmm15, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovups %%xmm13, (%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%xmm14, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
    codestream << "                         \"vmovups %%xmm15, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

void avx_store_1x3_sp_asm(std::stringstream& codestream, int ldc, bool alignC, std::string tPrefetch) {
  codestream << "                         \"vmovss %%xmm13, (%%r10)\\n\\t\"" << std::endl;
  codestream << "                         \"vmovss %%xmm14, " << ldc * 4 << "(%%r10)\\n\\t\"" << std::endl;
  codestream << "                         \"vmovss %%xmm15, " << (2 * ldc) * 4 << "(%%r10)\\n\\t\"" << std::endl;
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"prefetcht1 (%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << ldc * 4 << "(%%r12)\\n\\t\"" << std::endl;
    codestream << "                         \"prefetcht1 " << (2 * ldc) * 4 << "(%%r12)\\n\\t\"" << std::endl;
  }
}

