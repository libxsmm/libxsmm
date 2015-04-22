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

void avx_load_12xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_12xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (bAdd) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovapd " <<  (l_n * ldc)      * 8 << "(%%r10), %%ymm" << 7 + (3*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovapd " << ((l_n * ldc) + 4) * 8 << "(%%r10), %%ymm" << 8 + (3*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovapd " << ((l_n * ldc) + 8) * 8 << "(%%r10), %%ymm" << 9 + (3*l_n) << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovupd " <<  (l_n * ldc)      * 8 << "(%%r10), %%ymm" << 7 + (3*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovupd " << ((l_n * ldc) + 4) * 8 << "(%%r10), %%ymm" << 8 + (3*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovupd " << ((l_n * ldc) + 8) * 8 << "(%%r10), %%ymm" << 9 + (3*l_n) << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorpd %%ymm" << 7 + (3*l_n) << ", %%ymm" << 7 + (3*l_n) << ", %%ymm" << 7 + (3*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorpd %%ymm" << 8 + (3*l_n) << ", %%ymm" << 8 + (3*l_n) << ", %%ymm" << 8 + (3*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorpd %%ymm" << 9 + (3*l_n) << ", %%ymm" << 9 + (3*l_n) << ", %%ymm" << 9 + (3*l_n) << "\\n\\t\"" << std::endl;
    }
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 8 << "(%%r12)\\n\\t\"" << std::endl;
      codestream << "                         \"prefetcht1 " << ((l_n * ldc) + 8) * 8 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_load_8xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_8xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (bAdd) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovapd " <<  (l_n * ldc)      * 8 << "(%%r10), %%ymm" << 10 + (2*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovapd " << ((l_n * ldc) + 4) * 8 << "(%%r10), %%ymm" << 11 + (2*l_n) << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovupd " <<  (l_n * ldc)      * 8 << "(%%r10), %%ymm" << 10 + (2*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovupd " << ((l_n * ldc) + 4) * 8 << "(%%r10), %%ymm" << 11 + (2*l_n) << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorpd %%ymm" << 10 + (2*l_n) << ", %%ymm" << 10 + (2*l_n) << ", %%ymm" << 10 + (2*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorpd %%ymm" << 11 + (2*l_n) << ", %%ymm" << 11 + (2*l_n) << ", %%ymm" << 11 + (2*l_n) << "\\n\\t\"" << std::endl;
    }
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 8 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_load_4xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_4xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (bAdd) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovapd " <<  (l_n * ldc)      * 8 << "(%%r10), %%ymm" << 13 + l_n << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovupd " <<  (l_n * ldc)      * 8 << "(%%r10), %%ymm" << 13 + l_n << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorpd %%ymm" << 13 + l_n << ", %%ymm" << 13 + l_n << ", %%ymm" << 13 + l_n << "\\n\\t\"" << std::endl;
    }
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 8 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_load_2xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_2xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (bAdd) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovapd " <<  (l_n * ldc)      * 8 << "(%%r10), %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovupd " <<  (l_n * ldc)      * 8 << "(%%r10), %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorpd %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
    }
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 8 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_load_1xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, bool bAdd, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_load_1xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (bAdd) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovsd " <<  (l_n * ldc)      * 8 << "(%%r10), %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorpd %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << ", %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
    }
  }
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 8 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_12xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_12xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovapd %%ymm" << 7 + (3*l_n) << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd %%ymm" << 8 + (3*l_n) << ", " <<  ((l_n * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd %%ymm" << 9 + (3*l_n) << ", " <<  ((l_n * ldc) + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovupd %%ymm" << 7 + (3*l_n) << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd %%ymm" << 8 + (3*l_n) << ", " <<  ((l_n * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd %%ymm" << 9 + (3*l_n) << ", " <<  ((l_n * ldc) + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_8xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_8xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovapd %%ymm" << 10 + (2*l_n) << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd %%ymm" << 11 + (2*l_n) << ", " <<  ((l_n * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovupd %%ymm" << 10 + (2*l_n) << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd %%ymm" << 11 + (2*l_n) << ", " <<  ((l_n * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_4xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_4xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovapd %%ymm" << 13 + l_n << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovupd %%ymm" << 13 + l_n << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_2xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_2xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovapd %%xmm" << 13 + l_n << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovupd %%xmm" << 13 + l_n << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }
}

void avx_store_1xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx_store_1xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    codestream << "                         \"vmovsd %%xmm" << 13 + l_n << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
  }
}

