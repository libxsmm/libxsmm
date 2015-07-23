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

void avx512_init_registers_sp_asm(std::stringstream& codestream, std::string tPrefetch) {
  codestream << "    __asm__ __volatile__(\"movq %0, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"movq %1, %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"movq %2, %%r10\\n\\t\"" << std::endl;
  // B prefetch and prefetch within A
  if (    (tPrefetch.compare("BL2viaC") == 0) 
       || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
    codestream << "                         \"movq %3, %%r12\\n\\t\"" << std::endl;
  // A prefetch
  } else if (    (tPrefetch.compare("AL2jpst") == 0)
              || (tPrefetch.compare("AL2") == 0) ) {
    codestream << "                         \"movq %3, %%r11\\n\\t\"" << std::endl;
  // A and B prefetch
  } else if (    (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
              || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"movq %3, %%r11\\n\\t\"" << std::endl;
    codestream << "                         \"movq %4, %%r12\\n\\t\"" << std::endl;
  }
  codestream << "                         \"movq $0, %%r15\\n\\t\"" << std::endl;
//  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
}

void avx512_close_sp_asm(std::stringstream& codestream, int max_local_N, std::string tPrefetch) {
  // B prefetch and inside A prefetch
  if (    (tPrefetch.compare("BL2viaC") == 0) 
       || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
    codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"r8\",\"r9\",\"r10\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\"";
  // A jumpstart prefetch
  } else if (    (tPrefetch.compare("AL2jpst") == 0) 
              || (tPrefetch.compare("AL2") == 0) ) {
    codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"r8\",\"r9\",\"r10\",\"r11\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\"";
  // A and B prefetch
  } else if (    (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
              || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\"";
  } else {
    codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"r8\",\"r9\",\"r10\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\"";
  }
  // generate the clobber-list
  for (int n_local = 0; n_local < max_local_N; n_local++) {
    codestream << ",\"zmm" << 31-n_local << "\"";
  }
  codestream << ");" << std::endl;
}

void avx512_header_nloop_sp_asm(std::stringstream& codestream, int n_blocking) {
  codestream << "                         \"1" << n_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << n_blocking << ", %%r15\\n\\t\"" << std::endl;
}

void avx512_footer_nloop_sp_asm(std::stringstream& codestream, int n_blocking, int N, int M, int lda, int ldb, int ldc) {
  codestream << "                         \"addq $" << ((n_blocking)*ldc * 4) - (M * 4) << ", %%r10\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << n_blocking * ldb * 4 << ", %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << M * 4 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << N << ", %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"jl 1" << n_blocking << "b\\n\\t\"" << std::endl;
}

void avx512_header_kloop_sp_asm(std::stringstream& codestream, int m_blocking, int k_blocking) {
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"2" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << k_blocking << ", %%r13\\n\\t\"" << std::endl;
}

void avx512_footer_kloop_sp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << K * 4 << ", %%r8\\n\\t\"" << std::endl;
}

void avx512_footer_kloop_notdone_sp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
}

void avx512_header_mloop_sp_asm(std::stringstream& codestream, int m_blocking) {
  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"100" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << m_blocking << ", %%r14\\n\\t\"" << std::endl;
}

void avx512_footer_mloop_sp_asm(std::stringstream& codestream, int m_blocking, int K, int M_done, int lda, bool Kfullyunrolled, std::string tPrefetch) {
  codestream << "                         \"addq $" << m_blocking * 4 << ", %%r10\\n\\t\"" << std::endl;
  // B prefetch
  if (    (tPrefetch.compare("BL2viaC") == 0) 
       || (tPrefetch.compare("curAL2_BL2viaC") == 0)
       || (tPrefetch.compare("AL2_BL2viaC") == 0)
       || (tPrefetch.compare("AL2jpst_BL2viaC") == 0) ) {
    codestream << "                         \"addq $" << m_blocking * 4 << ", %%r12\\n\\t\"" << std::endl;
  }
  if (Kfullyunrolled == false) {
    codestream << "                         \"subq $" << (K * 4 * lda) - (m_blocking * 4) << ", %%r9\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"addq $" << (m_blocking * 4) << ", %%r9\\n\\t\"" << std::endl;
  }
  // A prefetch
  if (    (tPrefetch.compare("AL2_BL2viaC") == 0) 
       || (tPrefetch.compare("AL2") == 0) ) {
    if (Kfullyunrolled == false) {
      codestream << "                         \"subq $" << (K * 4 * lda) - (m_blocking * 8) << ", %%r11\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"addq $" << (m_blocking * 4) << ", %%r11\\n\\t\"" << std::endl;
    }
  }
  codestream << "                         \"cmpq $" << M_done << ", %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"jl 100" << m_blocking << "b\\n\\t\"" << std::endl;
}

void avx512_load_16xN_sp_asm(std::stringstream& codestream, int max_local_N, int ldc, bool alignC, bool bAdd, bool bUseMasking = false) {
  if (max_local_N > 30) {
    std::cout << " !!! ERROR, AVX-512, max_local_N > 30 !!! " << std::endl;
    exit(-1); 
  }

  if (bAdd) {
    if (alignC == true) {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << ldc * 4 * n_local << "(%%r10), %%zmm" << 31-n_local <<"\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << ldc * 4 * n_local << "(%%r10), %%zmm" << 31-n_local <<"{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    } else {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << ldc * 4 * n_local << "(%%r10), %%zmm" << 31-n_local <<"\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << ldc * 4 * n_local << "(%%r10), %%zmm" << 31-n_local <<"{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
  } else {
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vpxord %%zmm" << 31-n_local <<", %%zmm" << 31-n_local <<", %%zmm" << 31-n_local <<"\\n\\t\"" << std::endl;
    }
  }
}

void avx512_store_16xN_sp_asm(std::stringstream& codestream, int max_local_N, int ldc, bool alignC, std::string tPrefetch, bool bUseMasking = false) {
  if (max_local_N > 30) {
    std::cout << " !!! ERROR, AVX-512, max_local_N > 30 !!! " << std::endl;
    exit(-1); 
  }

  for (int n_local = 0; n_local < max_local_N; n_local++) {
    if (alignC == true) {
      if (bUseMasking == false) { 
        codestream << "                         \"vmovaps %%zmm" << 31-n_local <<", " << ldc * 4 * n_local << "(%%r10)\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps %%zmm" << 31-n_local <<", " << ldc * 4 * n_local << "(%%r10){{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vmovups %%zmm" << 31-n_local <<", " << ldc * 4 * n_local << "(%%r10)\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovups %%zmm" << 31-n_local <<", " << ldc * 4 * n_local << "(%%r10){{%%k1}}\\n\\t\"" << std::endl;
      }
    }
    // next B prefetch
    if (    (tPrefetch.compare("BL2viaC") == 0) 
         || (tPrefetch.compare("curAL2_BL2viaC") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0) 
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << ldc * 4 * n_local << "(%%r12)\\n\\t\"" << std::endl;                      
    }
  }
}

void avx512_kernel_16xN_sp_asm(std::stringstream& codestream, int max_local_N, int lda, int ldb, int ldc, bool alignA, int call, std::string tPrefetch, bool bUseMasking) {
  if (max_local_N > 30) {
    std::cout << " !!! ERROR, AVX-512, max_local_N > 30 !!! " << std::endl;
    exit(-1); 
  }

  if (call != (-1)) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovaps " << lda * 4 * call << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps " << lda * 4 * call << "(%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vmovups " << lda * 4 * call << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovups " << lda * 4 * call << "(%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
      }
    }
    // current A prefetch, next 8 rows for the current column
    if (    (tPrefetch.compare("curAL2") == 0) 
         || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 4 * call) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0)
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 4 * call) << "(%%r11)\\n\\t\"" << std::endl;
    }
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231ps " << (4 * call) + (ldb * 4 * n_local) << "(%%r8){{1to16}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  } else {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovaps (%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps (%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vmovups (%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovups (%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
      }
    }
    // current A prefetch, next 8 rows for the current column
    if (    (tPrefetch.compare("curAL2") == 0) 
         || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 64(%%r9)\\n\\t\"" << std::endl;
    }
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0) 
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 (%%r11)\\n\\t\"" << std::endl;
      codestream << "                         \"addq $" << lda * 4 << ", %%r11\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $" << lda * 4 << ", %%r9\\n\\t\"" << std::endl;
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231ps " << (ldb * 4 * n_local) << "(%%r8){{1to16}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }
}

void avx512_kernel_16xNx8_sp_asm(std::stringstream& codestream, int max_local_N, int lda, int ldb, int ldc, bool alignA, std::string tPrefetch, bool bUseMasking) {
  if (max_local_N > 30) {
    std::cout << " !!! ERROR, AVX-512, max_local_N > 30 !!! " << std::endl;
    exit(-1); 
  }

  for (int k = 0; k < 8; k++) {
    if (k == 0) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm1\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm1{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm1\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm1{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    } else if (k < 7){
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm" << (k+1)%2 << "\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm" << (k+1)%2 << "{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm" << (k+1)%2 << "\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm" << (k+1)%2 << "{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    // current A prefetch, next 8 rows for the current column
    if (    (tPrefetch.compare("curAL2") == 0) 
         || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 4 * k)+64 << "(%%r9)\\n\\t\"" << std::endl;
    }
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0)
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 4 * k) << "(%%r11)\\n\\t\"" << std::endl;
    }
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231ps " << (ldb * 4 * n_local) + (k * 4) << "(%%r8){{1to16}}, %%zmm" << k%2 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  }

  codestream << "                         \"addq $32, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << lda * 32 << ", %%r9\\n\\t\"" << std::endl;
  // next A prefetch "same" rows in "same" column, but in a different matrix 
  if (    (tPrefetch.compare("AL2jpst") == 0)
       || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
       || (tPrefetch.compare("AL2") == 0)
       || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
    codestream << "                         \"addq $" << lda * 32 << ", %%r11\\n\\t\"" << std::endl;
  }
}

void avx512_kernel_16x9xKfullyunrolled_indexed_sp_asm(std::stringstream& codestream, int max_local_K, int lda, int ldb, int ldc, bool alignA, std::string tPrefetch, bool bUseMasking) {
  codestream << "                         \"movq $" << ldb*4 << ", %%rax\\n\\t\"" << std::endl;
  codestream << "                         \"movq %%r8, %%rbx\\n\\t\"" << std::endl;
  codestream << "                         \"movq %%r8, %%rcx\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << ldb*4*3 << ", %%rbx\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << ldb*4*6 << ", %%rcx\\n\\t\"" << std::endl;
  
  int bdK = 0;
  int l_Kupdates = 0;

  for (int k = 0; k < max_local_K; k++) {
    if ( (k > 0) && (k%32 == 0) ) {
      codestream << "                         \"addq $128, %%r8\\n\\t\"" << std::endl;
      codestream << "                         \"addq $128, %%rbx\\n\\t\"" << std::endl;
      codestream << "                         \"addq $128, %%rcx\\n\\t\"" << std::endl;
      bdK = 0;
      l_Kupdates++;       
    }
    if ( k == 0 ) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm1\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm1{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm1\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm1{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    if ( k == 1 ) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm2\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+2) << "(%%r9), %%zmm3\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm2{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+2) << "(%%r9), %%zmm3{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm2\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+2) << "(%%r9), %%zmm3\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm2{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+2) << "(%%r9), %%zmm3{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    if ( k == 2 ) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * (k+2) << "(%%r9), %%zmm4\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+3) << "(%%r9), %%zmm5\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * (k+2) << "(%%r9), %%zmm4{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+3) << "(%%r9), %%zmm5{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * (k+2) << "(%%r9), %%zmm4\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+3) << "(%%r9), %%zmm5\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * (k+2) << "(%%r9), %%zmm4{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+3) << "(%%r9), %%zmm5{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    if ((k > 2) && (k < max_local_K-3)) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * (k+3) << "(%%r9), %%zmm" << (k+3)%6 << "\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * (k+3) << "(%%r9), %%zmm" << (k+3)%6 << "{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * (k+3) << "(%%r9), %%zmm" << (k+3)%6 << "\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * (k+3) << "(%%r9), %%zmm" << (k+3)%6 << "{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    // current A prefetch, next 8 rows for the current column
    if (    (tPrefetch.compare("curAL2") == 0) 
         || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 4 * k)+64 << "(%%r9)\\n\\t\"" << std::endl;
    }
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0)
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 4 * k) << "(%%r11)\\n\\t\"" << std::endl;
    }
    if (k%2 == 0) {
      for (int n_local = 0; n_local < 9; n_local++) {
        if (n_local == 0) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%r8){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 1) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%r8,%%rax,1){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 2) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%r8,%%rax,2){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 3) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%rbx){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 4) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%r8,%%rax,4){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 5) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%rbx,%%rax,2){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 6) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%rcx){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 7) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%rbx,%%rax,4){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 8) {
          codestream << "                         \"vfmadd231ps " << (bdK * 4) << "(%%r8,%%rax,8){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        }
      }
    } else {
      for (int n_local = 0; n_local < 9; n_local++) {
        std::string instr;
        if (k > 2) {
          instr = "vfmadd231ps ";
        } else {
          instr = "vmulps ";
        }
        if (n_local == 0) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%r8){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 1) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%r8,%%rax,1){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 2) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%r8,%%rax,2){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 3) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%rbx){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 4) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%r8,%%rax,4){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 5) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%rbx,%%rax,2){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 6) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%rcx){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 7) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%rbx,%%rax,4){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        } else if (n_local == 8) {
          codestream << "                         \"" << instr << (bdK * 4) << "(%%r8,%%rax,8){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-9-n_local << "\\n\\t\"" << std::endl;
        }
      }
    }
    bdK++;
  }
  for (int n_local = 0; n_local < 9; n_local++) {
    codestream << "                         \"vaddps %%zmm" << 31-n_local << ", %%zmm" << 31-9-n_local << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
  }
  // reset r8
  if (l_Kupdates > 0) {
    codestream << "                         \"subq $" << l_Kupdates*128 << ", %%r8\\n\\t\"" << std::endl;
  }

  if (    (tPrefetch.compare("AL2jpst") == 0)
       || (tPrefetch.compare("AL2jpst_BL2viaC") == 0) ) {
    codestream << "                         \"addq $" << lda * 4 * max_local_K << ", %%r11\\n\\t\"" << std::endl;
  }
}

void avx512_kernel_16x9x9fullyunrolled_indexed_sp_asm(std::stringstream& codestream, int max_local_K, int lda, int ldb, int ldc, bool alignA, std::string tPrefetch, bool bUseMasking) {
  codestream << "                         \"movq $" << ldb*4 << ", %%rax\\n\\t\"" << std::endl;
  codestream << "                         \"movq %%r8, %%rbx\\n\\t\"" << std::endl;
  codestream << "                         \"movq %%r8, %%rcx\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << ldb*4*3 << ", %%rbx\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << ldb*4*6 << ", %%rcx\\n\\t\"" << std::endl;
  
  for (int k = 0; k < max_local_K; k++) {
    if ( k == 0 ) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm1\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm1{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm1\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm1{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    if ( k == 1 ) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm2\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+2) << "(%%r9), %%zmm3\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * (k+1) << "(%%r9), %%zmm2{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+2) << "(%%r9), %%zmm3{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm2\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+2) << "(%%r9), %%zmm3\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * (k+1) << "(%%r9), %%zmm2{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+2) << "(%%r9), %%zmm3{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    if ( k == 2 ) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * (k+2) << "(%%r9), %%zmm4\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+3) << "(%%r9), %%zmm5\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * (k+2) << "(%%r9), %%zmm4{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovaps " << lda * 4 * (k+3) << "(%%r9), %%zmm5{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * (k+2) << "(%%r9), %%zmm4\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+3) << "(%%r9), %%zmm5\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * (k+2) << "(%%r9), %%zmm4{{%%k1}}{{z}}\\n\\t\"" << std::endl;
          codestream << "                         \"vmovups " << lda * 4 * (k+3) << "(%%r9), %%zmm5{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    if ((k > 2) && (k < max_local_K-3)) {
      if (alignA == true) {
        if (bUseMasking == false) {
          codestream << "                         \"vmovaps " << lda * 4 * (k+3) << "(%%r9), %%zmm" << (k+3)%6 << "\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovaps " << lda * 4 * (k+3) << "(%%r9), %%zmm" << (k+3)%6 << "{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      } else {
        if (bUseMasking == false) {
          codestream << "                         \"vmovups " << lda * 4 * (k+3) << "(%%r9), %%zmm" << (k+3)%6 << "\\n\\t\"" << std::endl;
        } else {
          codestream << "                         \"vmovups " << lda * 4 * (k+3) << "(%%r9), %%zmm" << (k+3)%6 << "{{%%k1}}{{z}}\\n\\t\"" << std::endl;
        }
      }
    }
    // current A prefetch, next 8 rows for the current column
    if (    (tPrefetch.compare("curAL2") == 0) 
         || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 4 * k)+64 << "(%%r9)\\n\\t\"" << std::endl;
    }
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0)
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 4 * k) << "(%%r11)\\n\\t\"" << std::endl;
    }
    for (int n_local = 0; n_local < 9; n_local++) {
      if (n_local == 0) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%r8){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      } else if (n_local == 1) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%r8,%%rax,1){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      } else if (n_local == 2) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%r8,%%rax,2){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      } else if (n_local == 3) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%rbx){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      } else if (n_local == 4) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%r8,%%rax,4){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      } else if (n_local == 5) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%rbx,%%rax,2){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      } else if (n_local == 6) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%rcx){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      } else if (n_local == 7) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%rbx,%%rax,4){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      } else if (n_local == 8) {
        codestream << "                         \"vfmadd231ps " << (k * 4) << "(%%r8,%%rax,8){{1to16}}, %%zmm" << k%6 << ", %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
      }
    }
  }

  if (    (tPrefetch.compare("AL2jpst") == 0)
       || (tPrefetch.compare("AL2jpst_BL2viaC") == 0) ) {
    codestream << "                         \"addq $" << lda * 4 * max_local_K << ", %%r11\\n\\t\"" << std::endl;
  }
}

bool avx512_generate_inner_k_loop_sp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch, bool bUseMasking) {
  int k_blocking = 8;
  int k_threshold = 8;
  bool KisFullyUnrolled = false;

  // Let's do something special for SeisSol high-order (N == 9 holds true)
  if ((K != 9) && (K >= 8) && (N == 9)) {
    avx512_kernel_16x9xKfullyunrolled_indexed_sp_asm(codestream, K, lda, ldb, ldc, alignA, tPrefetch, bUseMasking);
    KisFullyUnrolled = true;
  } else if ((K == 9) && (N == 9)) {
    avx512_kernel_16x9x9fullyunrolled_indexed_sp_asm(codestream, K, lda, ldb, ldc, alignA, tPrefetch, bUseMasking);
    KisFullyUnrolled = true;
  } else if (K % k_blocking == 0 && K >= k_threshold) {
    avx512_header_kloop_sp_asm(codestream, 16, k_blocking);
    if (k_blocking == 8) {
      avx512_kernel_16xNx8_sp_asm(codestream, N, lda, ldb, ldc, alignA, tPrefetch, bUseMasking);
    } else {
      std::cout << " !!! ERROR, AVX-512, k-blocking !!! " << std::endl;
      exit(-1);
    }
    avx512_footer_kloop_sp_asm(codestream, 16, K);
  } else {
    int max_blocked_K = (K/k_blocking)*k_blocking;
    if (max_blocked_K > 0 ) {
      avx512_header_kloop_sp_asm(codestream, 16, k_blocking);
      avx512_kernel_16xNx8_sp_asm(codestream, N, lda, ldb, ldc, alignA, tPrefetch, bUseMasking);
      avx512_footer_kloop_notdone_sp_asm(codestream, 16, max_blocked_K);
    }
    for (int i = max_blocked_K; i < K; i++) {
      avx512_kernel_16xN_sp_asm(codestream, N, lda, ldb, ldc, alignA, i-max_blocked_K, tPrefetch, bUseMasking);
    }
    // update r8 and r9
    codestream << "                         \"addq $" << lda * 4 * (K - max_blocked_K) << ", %%r9\\n\\t\"" << std::endl;
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0) 
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0)) {
      codestream << "                         \"addq $" << lda * 4 * (K - max_blocked_K) << ", %%r11\\n\\t\"" << std::endl;
    }
    if (max_blocked_K > 0 ) {
      codestream << "                         \"subq $" << max_blocked_K * 4 << ", %%r8\\n\\t\"" << std::endl;
    }
  }

  return KisFullyUnrolled;
}

void avx512_generate_inner_m_loop_sp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch) {
  int mDone = (M / 16) * 16;
  // multiples of 8 in M
  if (mDone > 0) {
    avx512_header_mloop_sp_asm(codestream, 16);
    avx512_load_16xN_sp_asm(codestream, N, ldc, alignC, bAdd, false);
    // innner loop over K
    bool KisFullyUnrolled = avx512_generate_inner_k_loop_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, tPrefetch, false);
    avx512_store_16xN_sp_asm(codestream, N, ldc, alignC, tPrefetch, false);
    avx512_footer_mloop_sp_asm(codestream, 16, K, mDone, lda, KisFullyUnrolled, tPrefetch);
  }

  // Remainder Handling using Masking, we are using M loop counter register as GP register
  // for the mask
  if (mDone != M) {
    switch(M - mDone)
    {
      case 1: 
        codestream << "                         \"movq $1, %%r14\\n\\t\"" << std::endl;
        break;
      case 2: 
        codestream << "                         \"movq $3, %%r14\\n\\t\"" << std::endl;
        break;
      case 3: 
        codestream << "                         \"movq $7, %%r14\\n\\t\"" << std::endl;
        break;
      case 4: 
        codestream << "                         \"movq $15, %%r14\\n\\t\"" << std::endl;
        break;
      case 5: 
        codestream << "                         \"movq $31, %%r14\\n\\t\"" << std::endl;
        break;
      case 6: 
        codestream << "                         \"movq $63, %%r14\\n\\t\"" << std::endl;
        break;
      case 7: 
        codestream << "                         \"movq $127, %%r14\\n\\t\"" << std::endl;
        break;
      case 8: 
        codestream << "                         \"movq $255, %%r14\\n\\t\"" << std::endl;
        break;       
      case 9: 
        codestream << "                         \"movq $511, %%r14\\n\\t\"" << std::endl;
        break;       
      case 10: 
        codestream << "                         \"movq $1023, %%r14\\n\\t\"" << std::endl;
        break;       
      case 11: 
        codestream << "                         \"movq $2047, %%r14\\n\\t\"" << std::endl;
        break;       
      case 12: 
        codestream << "                         \"movq $4095, %%r14\\n\\t\"" << std::endl;
        break;       
      case 13: 
        codestream << "                         \"movq $8191, %%r14\\n\\t\"" << std::endl;
        break;       
      case 14: 
        codestream << "                         \"movq $16383, %%r14\\n\\t\"" << std::endl;
        break;       
      case 15: 
        codestream << "                         \"movq $32767, %%r14\\n\\t\"" << std::endl;
    }
    codestream << "                         \"kmovw %%r14d, %%k1\\n\\t\"" << std::endl;
    avx512_load_16xN_sp_asm(codestream, N, ldc, alignC, bAdd, true);
    // innner loop over K
    avx512_generate_inner_k_loop_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, tPrefetch, true);
    avx512_store_16xN_sp_asm(codestream, N, ldc, alignC, tPrefetch, true);
    codestream << "                         \"addq $" << (M - mDone) * 4 << ", %%r10\\n\\t\"" << std::endl;
    codestream << "                         \"subq $" << (K * 4 * lda) - ( (M - mDone) * 4) << ", %%r9\\n\\t\"" << std::endl;
  }
}

void avx512_generate_kernel_sp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch) {
  int l_numberOfChunks = 1+((N-1)/30);
  int l_modulo = N%l_numberOfChunks;
  int l_n2 = N/l_numberOfChunks;
  int l_n1 = l_n2 + 1;
  int l_N2 = 0;
  int l_N1 = 0;
  if (l_n1 > 30) l_n1 = 30; // this just the case if N/l_numberOfChunks has no remainder

  for (int l_chunk = 0; l_chunk < l_numberOfChunks; l_chunk++) {
    if (l_chunk < l_modulo) {
      l_N1 += l_n1;
    } else {
      l_N2 += l_n2;
    }
  }
  
  std::cout << "N splitting of SP AVX512 Kernel: " << l_N1 << " " << l_N2 << " " << l_n1 << " " << l_n2 << std::endl;
  avx512_init_registers_sp_asm(codestream, tPrefetch);

  if (l_numberOfChunks == 1) {
    avx512_generate_inner_m_loop_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, tPrefetch);
  } else {
    if ((l_N2 > 0) && (l_N1 > 0)) {
      avx512_header_nloop_sp_asm(codestream, l_n1);
      avx512_generate_inner_m_loop_sp(codestream, lda, ldb, ldc, M, l_n1, K, alignA, alignC, bAdd, tPrefetch);
      avx512_footer_nloop_sp_asm(codestream, l_n1, l_N1, M, lda, ldb, ldc);

      avx512_header_nloop_sp_asm(codestream, l_n2);
      avx512_generate_inner_m_loop_sp(codestream, lda, ldb, ldc, M, l_n2, K, alignA, alignC, bAdd, tPrefetch);
      avx512_footer_nloop_sp_asm(codestream, l_n2, N, M, lda, ldb, ldc);
    } else if ((l_N2 > 0) && (l_N1 == 0)) {
      avx512_header_nloop_sp_asm(codestream, l_n2);
      avx512_generate_inner_m_loop_sp(codestream, lda, ldb, ldc, M, l_n2, K, alignA, alignC, bAdd, tPrefetch);
      avx512_footer_nloop_sp_asm(codestream, l_n2, N, M, lda, ldb, ldc);
    } else {
      std::cout << " !!! ERROR, AVX512 n-blocking !!! " << std::endl;
      exit(-1);
    }
  }
  
  if (N == 9) {
    avx512_close_sp_asm(codestream, 30, tPrefetch);
  } else {
    avx512_close_sp_asm(codestream, std::max<int>(l_n1, l_n2), tPrefetch);
  }
}

