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

void avx512knc_init_registers_dp_asm(std::stringstream& codestream) {
  codestream << "    __asm__ __volatile__(\"movq %0, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"movq %1, %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"movq %2, %%r10\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r15\\n\\t\"" << std::endl;
}

void avx512knc_close_dp_asm(std::stringstream& codestream, int max_local_N) {
  codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C) : \"k1\",\"r8\",\"r9\",\"r10\",\"r13\",\"r14\",\"r15\",\"zmm0\"";
  // generate the clobber-list
  for (int n_local = 0; n_local < max_local_N; n_local++) {
    codestream << ",\"zmm" << 31-n_local << "\"";
  }
  codestream << ");" << std::endl;
}

void avx512knc_header_nloop_dp_asm(std::stringstream& codestream, int n_blocking) {
  codestream << "                         \"1" << n_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << n_blocking << ", %%r15\\n\\t\"" << std::endl;
}

void avx512knc_footer_nloop_dp_asm(std::stringstream& codestream, int n_blocking, int N, int M, int lda, int ldb, int ldc) {
  codestream << "                         \"addq $" << ((n_blocking)*ldc * 8) - (M * 8) << ", %%r10\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << n_blocking * ldb * 8 << ", %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << M * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << N << ", %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"jl 1" << n_blocking << "b\\n\\t\"" << std::endl;
}

void avx512knc_header_kloop_dp_asm(std::stringstream& codestream, int m_blocking, int k_blocking) {
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"2" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << k_blocking << ", %%r13\\n\\t\"" << std::endl;
}

void avx512knc_footer_kloop_dp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << K * 8 << ", %%r8\\n\\t\"" << std::endl;
}

void avx512knc_footer_kloop_notdone_dp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
}

void avx512knc_header_mloop_dp_asm(std::stringstream& codestream, int m_blocking) {
  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"100" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << m_blocking << ", %%r14\\n\\t\"" << std::endl;
}

void avx512knc_footer_mloop_dp_asm(std::stringstream& codestream, int m_blocking, int K, int M_done, int lda) {
  codestream << "                         \"addq $" << m_blocking * 8 << ", %%r10\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << (K * 8 * lda) - (m_blocking * 8) << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << M_done << ", %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"jl 100" << m_blocking << "b\\n\\t\"" << std::endl;
}

void avx512knc_load_8xN_dp_asm(std::stringstream& codestream, int max_local_N, int ldc, bool alignC, bool bAdd, bool bUseMasking) {
  if (bAdd) {
    if (alignC == true) {
      if (bUseMasking == false) {
        for (int n_local = 0; n_local < max_local_N; n_local++) {
          codestream << "                         \"vmovapd " << ldc * 8 * n_local << "(%%r10), %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
          codestream << "                         \"vprefetch1 " << (ldc * 8 * n_local) + 64 << "(%%r10)\\n\\t\"" << std::endl;
        }
      } else {
        for (int n_local = 0; n_local < max_local_N; n_local++) {
          codestream << "                         \"vmovapd " << ldc * 8 * n_local << "(%%r10), %%zmm" << 31-n_local << "{{%%k1}}\\n\\t\"" << std::endl;
          codestream << "                         \"vprefetch1 " << (ldc * 8 * n_local) + 64 << "(%%r10)\\n\\t\"" << std::endl;
        }
      }
    } else {
      if (bUseMasking == false) {
        for (int n_local = 0; n_local < max_local_N; n_local++) {
          codestream << "                         \"vloadunpacklpd " << ldc * 8 * n_local << "(%%r10), %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
          codestream << "                         \"vloadunpackhpd " << (ldc * 8 * n_local)+64 << "(%%r10), %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        }
      } else {
        for (int n_local = 0; n_local < max_local_N; n_local++) {
          codestream << "                         \"vloadunpacklpd " << ldc * 8 * n_local << "(%%r10), %%zmm" << 31-n_local << "{{%%k1}}\\n\\t\"" << std::endl;
          codestream << "                         \"vloadunpackhpd " << (ldc * 8 * n_local)+64 << "(%%r10), %%zmm" << 31-n_local << "{{%%k1}}\\n\\t\"" << std::endl;
        }
      }
    }
  } else {
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vpxord %%zmm" << 31-n_local <<", %%zmm" << 31-n_local <<", %%zmm" << 31-n_local <<"\\n\\t\"" << std::endl;
      codestream << "                         \"vprefetch1 " << ldc * 8 * n_local << "(%%r10)\\n\\t\"" << std::endl;
    }
  }
}

void avx512knc_store_8xN_dp_asm(std::stringstream& codestream, int max_local_N, int ldc, bool alignC, bool bUseMasking) {
  if (alignC == true) {
    if (bUseMasking == false) {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        codestream << "                         \"vmovapd %%zmm" << 31-n_local <<", " << ldc * 8 * n_local << "(%%r10)\\n\\t\"" << std::endl;
      }
    } else {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        codestream << "                         \"vmovapd %%zmm" << 31-n_local <<", " << ldc * 8 * n_local << "(%%r10){{%%k1}}\\n\\t\"" << std::endl;
      }
    }
  } else {
    if (bUseMasking == false) {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        codestream << "                         \"vpackstorelpd %%zmm" << 31-n_local <<", " << ldc * 8 * n_local << "(%%r10)\\n\\t\"" << std::endl;
        codestream << "                         \"vpackstorehpd %%zmm" << 31-n_local <<", " << (ldc * 8 * n_local)+64 << "(%%r10)\\n\\t\"" << std::endl;
      }
    } else {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        codestream << "                         \"vpackstorelpd %%zmm" << 31-n_local <<", " << ldc * 8 * n_local << "(%%r10){{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vpackstorehpd %%zmm" << 31-n_local <<", " << (ldc * 8 * n_local)+64 << "(%%r10){{%%k1}}\\n\\t\"" << std::endl;
      }
    }
  }
}

void avx512knc_kernel_8xN_dp_asm(std::stringstream& codestream, int max_local_N, int lda, int ldb, int ldc, bool alignA, bool alignC, int call, bool bUseMasking) {
  if (call != (-1)) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovapd " << lda * 8 * call << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovapd " << lda * 8 * call << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * call << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * call)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * call << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * call)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231pd " << (8 * call) + (ldb * 8 * n_local) << "(%%r8){{1to8}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  } else {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovapd (%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovapd (%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklpd (%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd 64(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklpd (%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd 64(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }
    codestream << "                         \"addq $" << lda * 8 << ", %%r9\\n\\t\"" << std::endl;
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231pd " << (ldb * 8 * n_local) << "(%%r8){{1to8}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }
}

// saved in case of SeisSol performance regression
#if 0
void avx512knc_kernel_8x9x8_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, bool blast, bool bUseMasking) {
  for (int k = 0; k < 8; k++) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovapd " << lda * 8 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovapd " << lda * 8 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * k)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * k)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }

    for (int n_local = 0; n_local < 9 /*max_local_N*/; n_local++) {
      if (n_local == 0) {
        if (k == 0) {
          codestream << "                         \"vprefetch0 " << lda * 8 * 1 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 1) {
          codestream << "                         \"vprefetch0 " << lda * 8 * 5 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 2) {
          codestream << "                         \"vprefetch1 " << 64 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 3) {
          codestream << "                         \"vprefetch1 " << (lda * 8 * 4) + 64 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 4) {
          codestream << "                         \"vprefetch0 " << 64 << "(%%r8)\\n\\t\"" << std::endl;
        } else if (k == 5) {
          codestream << "                         \"vprefetch0 " << (ldb * 8 * 4) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        } else if (k == 6) {
          codestream << "                         \"vprefetch0 " << (ldb * 8 * 8) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        }
      } else if (n_local == 1) {
        if (k == 0) {
          codestream << "                         \"vprefetch0 " << lda * 8 * 2 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 1) {
          codestream << "                         \"vprefetch0 " << lda * 8 * 6 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 2) {
          codestream << "                         \"vprefetch1 " << (lda * 8) + 64 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 3) {
          codestream << "                         \"vprefetch1 " << (lda * 8 * 5) + 64 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 4) {
          codestream << "                         \"vprefetch0 " << (ldb * 8) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        } else if (k == 5) {
          codestream << "                         \"vprefetch0 " << (ldb * 8 * 5) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        }
      } else if (n_local == 2) {
        if (k == 0) {
          codestream << "                         \"vprefetch0 " << lda * 8 * 3 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 1) {
          codestream << "                         \"vprefetch0 " << lda * 8 * 7 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 2) {
          codestream << "                         \"vprefetch1 " << (lda * 8 * 2) + 64 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 3) {
          codestream << "                         \"vprefetch1 " << (lda * 8 * 6) + 64 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 4) {
          codestream << "                         \"vprefetch0 " << (ldb * 8 * 2) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        } else if (k == 5) {
          codestream << "                         \"vprefetch0 " << (ldb * 8 * 6) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        }
      } else if (n_local == 3) {
        if (k == 0) {
          codestream << "                         \"vprefetch0 " << lda * 8 * 4 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 1) {
          codestream << "                         \"vprefetch0 " << lda * 8 * 8 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 2) {
          codestream << "                         \"vprefetch1 " << (lda * 8 * 3) + 64 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 3) {
          codestream << "                         \"vprefetch1 " << (lda * 8 * 7) + 64 << "(%%r9)\\n\\t\"" << std::endl;
        } else if (k == 4) {
          codestream << "                         \"vprefetch0 " << (ldb * 8 * 3) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        } else if (k == 5) {
          codestream << "                         \"vprefetch0 " << (ldb * 8 * 7) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        }
      }
      codestream << "                         \"vfmadd231pd " << (ldb * 8 * n_local) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  }

  codestream << "                         \"addq $64, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << lda * 64 << ", %%r9\\n\\t\"" << std::endl;
}
#endif

void avx512knc_kernel_8xNx8_dp_asm(std::stringstream& codestream, int max_local_N, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, bool blast, bool bUseMasking) {
  int n_B_prefetches = 0;
  for (int k = 0; k < 8; k++) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovapd " << lda * 8 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovapd " << lda * 8 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * k)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * k)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }

    for (int n_local = 0; n_local < max_local_N; n_local++) {
      if (n_local == 0) {
        // most important prefetch, next A
        codestream << "                         \"vprefetch0 " << lda * 8 * (1+k) << "(%%r9)\\n\\t\"" << std::endl;
      } if (n_local == 1) {
        // second most important prefetch, next rows of A into L2
        codestream << "                         \"vprefetch1 " << (lda * 8 * k) + 64 << "(%%r9)\\n\\t\"" << std::endl;
      } if ((n_local == 2) && (n_B_prefetches < max_local_N)) {
        // prefetch next B
        codestream << "                         \"vprefetch0 " << (ldb * 8 * n_B_prefetches) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        n_B_prefetches++;
      }
      codestream << "                         \"vfmadd231pd " << (ldb * 8 * n_local) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  }

  codestream << "                         \"addq $64, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << lda * 64 << ", %%r9\\n\\t\"" << std::endl;
}

// saved in case of SeisSol performance regression
#if 0
void avx512knc_kernel_8x9x9_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, bool blast, bool bUseMasking) {
  for (int k = 0; k < 9; k++) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovapd " << lda * 8 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovapd " << lda * 8 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * k)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * k)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 1 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 5 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 8 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 8 * 3) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 4) {
      codestream << "                         \"vprefetch1 " << (lda * 8 * 7) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231pd " << k * 8 << "(%%r8){{1to8}}, %%zmm0, %%zmm31\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 2 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 6 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 8 * 4) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 4) {
      codestream << "                         \"vprefetch1 " << (lda * 8 * 8) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231pd " << (ldb * 8) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm30\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 3 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 7 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << (lda * 8) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 8 * 5) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231pd " << (ldb * 16) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm29\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 4 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 8 * 8 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << (lda * 8 * 2) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 8 * 6) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231pd " << (ldb * 24) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm28\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231pd " << (ldb * 32) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm27\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231pd " << (ldb * 40) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm26\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231pd " << (ldb * 48) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm25\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231pd " << (ldb * 56) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm24\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231pd " << (ldb * 64) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm23\\n\\t\"" << std::endl;
  }

  codestream << "                         \"addq $" << lda * 72 << ", %%r9\\n\\t\"" << std::endl;
}
#endif

void avx512knc_kernel_8xNx9_dp_asm(std::stringstream& codestream, int max_local_N, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, bool blast, bool bUseMasking) {
  for (int k = 0; k < 9; k++) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovapd " << lda * 8 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovapd " << lda * 8 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * k)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklpd " << lda * 8 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhpd " << (lda * 8 * k)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }

    for (int n_local = 0; n_local < max_local_N; n_local++) {
      if (n_local == 0) {
        // most important prefetch, next A
        codestream << "                         \"vprefetch0 " << lda * 8 * (1+k) << "(%%r9)\\n\\t\"" << std::endl;
      } if (n_local == 1) {
        // second most important prefetch, next rows of A into L2
        codestream << "                         \"vprefetch1 " << (lda * 8 * k) + 64 << "(%%r9)\\n\\t\"" << std::endl;
      }
      codestream << "                         \"vfmadd231pd " << (ldb * 8 * n_local) + (k * 8) << "(%%r8){{1to8}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  }

  codestream << "                         \"addq $" << lda * 72 << ", %%r9\\n\\t\"" << std::endl;
}


void avx512knc_generate_inner_k_loop_dp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, bool bUseMasking) {
  int k_blocking = 8;

  if (K % k_blocking == 0 ) {
    if (k_blocking == 8) {
      avx512knc_header_kloop_dp_asm(codestream, 8, k_blocking);
      avx512knc_kernel_8xNx8_dp_asm(codestream, N, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
      //avx512knc_kernel_8x9x8_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
      avx512knc_footer_kloop_dp_asm(codestream, 8, K);
    } else {
      std::cout << " !!! ERROR, MIC k-blocking !!! " << std::endl;
      exit(-1);
    }
  } else {
    // SeisSol special kernel
    if ( K == 9 ) {
      avx512knc_kernel_8xNx9_dp_asm(codestream, N, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
      //avx512knc_kernel_8x9x9_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
    } else {
      int max_blocked_K = (K/k_blocking)*k_blocking;
      if (max_blocked_K > 0 ) {
        avx512knc_header_kloop_dp_asm(codestream, 8, k_blocking);
        avx512knc_kernel_8xNx8_dp_asm(codestream, N, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
        //avx512knc_kernel_8x9x8_dp_asm(codestream, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
        avx512knc_footer_kloop_notdone_dp_asm(codestream, 8, max_blocked_K);
      }
      for (int i = max_blocked_K; i < K; i++) {
        avx512knc_kernel_8xN_dp_asm(codestream, N, lda, ldb, ldc, alignA, alignC, i-max_blocked_K, bUseMasking);
      }
      // update r8 and r9
      codestream << "                         \"addq $" << lda * 8 * (K - max_blocked_K) << ", %%r9\\n\\t\"" << std::endl;
      if (max_blocked_K > 0 ) {
        codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
      }
    }
  }  
}

void avx512knc_generate_inner_m_loop_dp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd) {
  int mDone = (M / 8) * 8;
  // multiples of 8 in M
  if (mDone > 0) {
    avx512knc_header_mloop_dp_asm(codestream, 8);
    avx512knc_load_8xN_dp_asm(codestream, N, ldc, alignC, bAdd, false);
    avx512knc_generate_inner_k_loop_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, false);
    avx512knc_store_8xN_dp_asm(codestream, N, ldc, alignC, false);
    avx512knc_footer_mloop_dp_asm(codestream, 8, K, mDone, lda);
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
    }
    codestream << "                         \"kmov %%r14d, %%k1\\n\\t\"" << std::endl;
    avx512knc_load_8xN_dp_asm(codestream, N, ldc, alignC, bAdd, true);
    // innner loop over K
    avx512knc_generate_inner_k_loop_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, true);
    avx512knc_store_8xN_dp_asm(codestream, N, ldc, alignC, true);
    codestream << "                         \"addq $" << (M - mDone) * 8 << ", %%r10\\n\\t\"" << std::endl;
    codestream << "                         \"subq $" << (K * 8 * lda) - ( (M - mDone) * 8) << ", %%r9\\n\\t\"" << std::endl;
  }
}

void avx512knc_generate_kernel_dp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd) {
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
  
  std::cout << "N splitting of DP MIC Kernel: " << l_N1 << " " << l_N2 << " " << l_n1 << " " << l_n2 << std::endl;
  avx512knc_init_registers_dp_asm(codestream);

  if (l_numberOfChunks == 1) {
    avx512knc_generate_inner_m_loop_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd);
  } else {
    if ((l_N2 > 0) && (l_N1 > 0)) {
      avx512knc_header_nloop_dp_asm(codestream, l_n1);
      avx512knc_generate_inner_m_loop_dp(codestream, lda, ldb, ldc, M, l_n1, K, alignA, alignC, bAdd);
      avx512knc_footer_nloop_dp_asm(codestream, l_n1, l_N1, M, lda, ldb, ldc);

      avx512knc_header_nloop_dp_asm(codestream, l_n2);
      avx512knc_generate_inner_m_loop_dp(codestream, lda, ldb, ldc, M, l_n2, K, alignA, alignC, bAdd);
      avx512knc_footer_nloop_dp_asm(codestream, l_n2, N, M, lda, ldb, ldc);
    } else if ((l_N2 > 0) && (l_N1 == 0)) {
      avx512knc_header_nloop_dp_asm(codestream, l_n2);
      avx512knc_generate_inner_m_loop_dp(codestream, lda, ldb, ldc, M, l_n2, K, alignA, alignC, bAdd);
      avx512knc_footer_nloop_dp_asm(codestream, l_n2, N, M, lda, ldb, ldc);
    } else {
      std::cout << " !!! ERROR, MIC n-blocking !!! " << std::endl;
      exit(-1);
    }
  }

  avx512knc_close_dp_asm(codestream, std::max<int>(l_n1, l_n2));
}

