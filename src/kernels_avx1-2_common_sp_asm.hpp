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

void avx_load_24xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int i_beta, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_24xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (i_beta == 1) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovaps " <<  (l_n * ldc)      * 4 << "(%%r10), %%ymm" << 7 + (3*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovaps " << ((l_n * ldc) + 8) * 4 << "(%%r10), %%ymm" << 8 + (3*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovaps " << ((l_n * ldc) +16) * 4 << "(%%r10), %%ymm" << 9 + (3*l_n) << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovups " <<  (l_n * ldc)      * 4 << "(%%r10), %%ymm" << 7 + (3*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovups " << ((l_n * ldc) + 8) * 4 << "(%%r10), %%ymm" << 8 + (3*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovups " << ((l_n * ldc) +16) * 4 << "(%%r10), %%ymm" << 9 + (3*l_n) << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorps %%ymm" << 7 + (3*l_n) << ", %%ymm" << 7 + (3*l_n) << ", %%ymm" << 7 + (3*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorps %%ymm" << 8 + (3*l_n) << ", %%ymm" << 8 + (3*l_n) << ", %%ymm" << 8 + (3*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorps %%ymm" << 9 + (3*l_n) << ", %%ymm" << 9 + (3*l_n) << ", %%ymm" << 9 + (3*l_n) << "\\n\\t\"" << std::endl;
    }
  }
}

void avx_load_16xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int i_beta, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_16xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (i_beta == 1) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovaps " <<  (l_n * ldc)      * 4 << "(%%r10), %%ymm" << 10 + (2*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovaps " << ((l_n * ldc) + 8) * 4 << "(%%r10), %%ymm" << 11 + (2*l_n) << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovups " <<  (l_n * ldc)      * 4 << "(%%r10), %%ymm" << 10 + (2*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovups " << ((l_n * ldc) + 8) * 4 << "(%%r10), %%ymm" << 11 + (2*l_n) << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorps %%ymm" << 10 + (2*l_n) << ", %%ymm" << 10 + (2*l_n) << ", %%ymm" << 10 + (2*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorps %%ymm" << 11 + (2*l_n) << ", %%ymm" << 11 + (2*l_n) << ", %%ymm" << 11 + (2*l_n) << "\\n\\t\"" << std::endl;
    }
  }
}

void avx_load_8xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int i_beta, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_8xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (i_beta == 1) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovaps " <<  (l_n * ldc)      * 4 << "(%%r10), %%ymm" << 13 + l_n << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovups " <<  (l_n * ldc)      * 4 << "(%%r10), %%ymm" << 13 + l_n << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorps %%ymm" << 13 + l_n << ", %%ymm" << 13 + l_n << ", %%ymm" << 13 + l_n << "\\n\\t\"" << std::endl;
    }
  }
}

void avx_load_4xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int i_beta, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_4xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (i_beta == 1) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovaps " <<  (l_n * ldc)      * 4 << "(%%r10), %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovups " <<  (l_n * ldc)      * 4 << "(%%r10), %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorps %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
    }
  }
}

void avx_load_1xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int i_beta, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_1xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (i_beta == 1) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovss " <<  (l_n * ldc)      * 4 << "(%%r10), %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorps %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_24xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_24xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovaps %%ymm" << 7 + (3*l_n) << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps %%ymm" << 8 + (3*l_n) << ", " <<  ((l_n * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps %%ymm" << 9 + (3*l_n) << ", " <<  ((l_n * ldc) +16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovups %%ymm" << 7 + (3*l_n) << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups %%ymm" << 8 + (3*l_n) << ", " <<  ((l_n * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups %%ymm" << 9 + (3*l_n) << ", " <<  ((l_n * ldc) +16) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }

  if ( (tPrefetch.compare("BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)       * 4 << "(%%r12)\\n\\t\"" << std::endl;
      codestream << "                         \"prefetcht1 " << ((l_n * ldc) + 16) * 4 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_16xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_16xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovaps %%ymm" << 10 + (2*l_n) << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovaps %%ymm" << 11 + (2*l_n) << ", " <<  ((l_n * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovups %%ymm" << 10 + (2*l_n) << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovups %%ymm" << 11 + (2*l_n) << ", " <<  ((l_n * ldc) + 8) * 4 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }

  if ( (tPrefetch.compare("BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 4 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_8xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_8xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovaps %%ymm" << 13 + l_n << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovups %%ymm" << 13 + l_n << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }

  if ( (tPrefetch.compare("BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 4 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_4xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_4xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovaps %%xmm" << 13 + l_n << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovups %%xmm" << 13 + l_n << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }

  if ( (tPrefetch.compare("BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 4 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_1xN_sp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_1xN_sp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    codestream << "                         \"vmovss %%xmm" << 13 + l_n << ", " <<   (l_n * ldc)      * 4 << "(%%r10)\\n\\t\"" << std::endl;
  }

  if ( (tPrefetch.compare("BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 4 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

