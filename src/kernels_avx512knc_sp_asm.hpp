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

void avx512knc_init_registers_sp_asm(std::stringstream& codestream) {
  codestream << "    __asm__ __volatile__(\"movq %0, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"movq %1, %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"movq %2, %%r10\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
}

void avx512knc_close_sp_asm(std::stringstream& codestream, int max_local_N) {
  codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C) : \"k1\",\"r8\",\"r9\",\"r10\",\"r13\",\"r14\",\"zmm0\"";
  // generate the clobber-list
  for (int n_local = 0; n_local < max_local_N; n_local++) {
    codestream << ",\"zmm" << 31-n_local << "\"";
  }
  codestream << ");" << std::endl;
}

void avx512knc_header_kloop_sp_asm(std::stringstream& codestream, int m_blocking, int k_blocking) {
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"2" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << k_blocking << ", %%r13\\n\\t\"" << std::endl;
}

void avx512knc_footer_kloop_sp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << K * 4 << ", %%r8\\n\\t\"" << std::endl;
}

void avx512knc_footer_kloop_notdone_sp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
}

void avx512knc_header_mloop_sp_asm(std::stringstream& codestream, int m_blocking) {
  codestream << "                         \"100" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << m_blocking << ", %%r14\\n\\t\"" << std::endl;
}

void avx512knc_footer_mloop_sp_asm(std::stringstream& codestream, int m_blocking, int K, int M_done, int lda) {
  codestream << "                         \"addq $" << m_blocking * 4 << ", %%r10\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << (K * 4 * lda) - (m_blocking * 4) << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << M_done << ", %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"jl 100" << m_blocking << "b\\n\\t\"" << std::endl;
}

void avx512knc_load_16xN_sp_asm(std::stringstream& codestream, int max_local_N, int ldc, bool alignC, bool bAdd, bool bUseMasking) {
  if (bAdd) {
    if (alignC == true) {
      if (bUseMasking == false) {
        for (int n_local = 0; n_local < max_local_N; n_local++) {
          codestream << "                         \"vmovaps " << ldc * 4 * n_local << "(%%r10), %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
          codestream << "                         \"vprefetch1 " << (ldc * 4 * n_local) + 64 << "(%%r10)\\n\\t\"" << std::endl;
        }
      } else {
        for (int n_local = 0; n_local < max_local_N; n_local++) {
          codestream << "                         \"vmovaps " << ldc * 4 * n_local << "(%%r10), %%zmm" << 31-n_local << "{{%%k1}}\\n\\t\"" << std::endl;
          codestream << "                         \"vprefetch1 " << (ldc * 4 * n_local) + 64 << "(%%r10)\\n\\t\"" << std::endl;
        }
      }
    } else {
      if (bUseMasking == false) {
        for (int n_local = 0; n_local < max_local_N; n_local++) {
          codestream << "                         \"vloadunpacklps " << ldc * 4 * n_local << "(%%r10), %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
          codestream << "                         \"vloadunpackhps " << (ldc * 4 * n_local)+64 << "(%%r10), %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
        }
      } else {
        for (int n_local = 0; n_local < max_local_N; n_local++) {
          codestream << "                         \"vloadunpacklps " << ldc * 4 * n_local << "(%%r10), %%zmm" << 31-n_local << "{{%%k1}}\\n\\t\"" << std::endl;
          codestream << "                         \"vloadunpackhps " << (ldc * 4 * n_local)+64 << "(%%r10), %%zmm" << 31-n_local << "{{%%k1}}\\n\\t\"" << std::endl;
        }
      }
    }
  } else {
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vpxord %%zmm" << 31-n_local <<", %%zmm" << 31-n_local <<", %%zmm" << 31-n_local <<"\\n\\t\"" << std::endl;
      codestream << "                         \"vprefetch1 " << ldc * 4 * n_local << "(%%r10)\\n\\t\"" << std::endl;
    }
  }
}

void avx512knc_store_16xN_sp_asm(std::stringstream& codestream, int max_local_N, int ldc, bool alignC, bool bUseMasking) {
  if (alignC == true) {
    if (bUseMasking == false) {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        codestream << "                         \"vmovaps %%zmm" << 31-n_local <<", " << ldc * 4 * n_local << "(%%r10)\\n\\t\"" << std::endl;
      }
    } else {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        codestream << "                         \"vmovaps %%zmm" << 31-n_local <<", " << ldc * 4 * n_local << "(%%r10){{%%k1}}\\n\\t\"" << std::endl;
      }
    }
  } else {
    if (bUseMasking == false) {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        codestream << "                         \"vpackstorelps %%zmm" << 31-n_local <<", " << ldc * 4 * n_local << "(%%r10)\\n\\t\"" << std::endl;
        codestream << "                         \"vpackstorehps %%zmm" << 31-n_local <<", " << (ldc * 4 * n_local)+64 << "(%%r10)\\n\\t\"" << std::endl;
      }
    } else {
      for (int n_local = 0; n_local < max_local_N; n_local++) {
        codestream << "                         \"vpackstorelps %%zmm" << 31-n_local <<", " << ldc * 4 * n_local << "(%%r10){{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vpackstorehps %%zmm" << 31-n_local <<", " << (ldc * 4 * n_local)+64 << "(%%r10){{%%k1}}\\n\\t\"" << std::endl;
      }
    }
  }
}

void avx512knc_kernel_16xN_sp_asm(std::stringstream& codestream, int max_local_N, int lda, int ldb, int ldc, bool alignA, bool alignC, int call, bool bUseMasking) {
  if (call != (-1)) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovaps " << lda * 4 * call << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps " << lda * 4 * call << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklps " << lda * 4 * call << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * call)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklps " << lda * 4 * call << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * call)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231ps " << (4 * call) + (ldb * 4 * n_local) << "(%%r8){{1to16}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  } else {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovaps (%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps (%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklps (%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps 64(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklps (%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps 64(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }
    codestream << "                         \"addq $" << lda * 4 << ", %%r9\\n\\t\"" << std::endl;
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231ps " << (ldb * 4 * n_local) << "(%%r8){{1to16}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $4, %%r8\\n\\t\"" << std::endl;
  }
}

void avx512knc_kernel_16xNx8_sp_asm(std::stringstream& codestream, int max_local_N, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, bool blast, bool bUseMasking) {
  int n_B_prefetches = 0;
  for (int k = 0; k < 8; k++) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * k)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * k)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }

    for (int n_local = 0; n_local < max_local_N; n_local++) {
      if (n_local == 0) {
        // most important prefetch, next A
        codestream << "                         \"vprefetch0 " << lda * 4 * (1+k) << "(%%r9)\\n\\t\"" << std::endl;
      } if (n_local == 1) {
        // second most important prefetch, next rows of A into L2
        codestream << "                         \"vprefetch1 " << (lda * 4 * k) + 64 << "(%%r9)\\n\\t\"" << std::endl;
      } if ((n_local == 2) && (n_B_prefetches < max_local_N)) {
        // prefetch next B
        codestream << "                         \"vprefetch0 " << (ldb * 4 * n_B_prefetches) + 64 << "(%%r8)\\n\\t\"" << std::endl;
        n_B_prefetches++;
      }
      codestream << "                         \"vfmadd231ps " << (ldb * 4 * n_local) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  }

  codestream << "                         \"addq $32, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << lda * 32 << ", %%r9\\n\\t\"" << std::endl;
}

void avx512knc_kernel_16xNx9_sp_asm(std::stringstream& codestream, int max_local_N, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, bool blast, bool bUseMasking) {
  for (int k = 0; k < 9; k++) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * k)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * k)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }

    for (int n_local = 0; n_local < max_local_N; n_local++) {
      if (n_local == 0) {
        // most important prefetch, next A
        codestream << "                         \"vprefetch0 " << lda * 4 * (1+k) << "(%%r9)\\n\\t\"" << std::endl;
      } if (n_local == 1) {
        // second most important prefetch, next rows of A into L2
        codestream << "                         \"vprefetch1 " << (lda * 4 * k) + 64 << "(%%r9)\\n\\t\"" << std::endl;
      }
      codestream << "                         \"vfmadd231ps " << (ldb * 4 * n_local) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
  }

  codestream << "                         \"addq $" << lda * 36 << ", %%r9\\n\\t\"" << std::endl;
}

// saved in case of SeisSol performance regression
#if 0
void avx512knc_kernel_16x9x8_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, bool blast, bool bUseMasking) {
  for (int k = 0; k < 8; k++) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * k)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * k)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 1 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 5 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 4) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 4) {
      codestream << "                         \"vprefetch0 " << 64 << "(%%r8)\\n\\t\"" << std::endl;
    } else if (k == 5) {
      codestream << "                         \"vprefetch0 " << (ldb * 4 * 4) + 64 << "(%%r8)\\n\\t\"" << std::endl;
    } else if (k == 6) {
      codestream << "                         \"vprefetch0 " << (ldb * 4 * 8) + 64 << "(%%r8)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231ps " << k * 4 << "(%%r8){{1to16}}, %%zmm0, %%zmm31\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 2 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 6 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << (lda * 4) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 5) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 4) {
      codestream << "                         \"vprefetch0 " << (ldb * 4) + 64 << "(%%r8)\\n\\t\"" << std::endl;
    } else if (k == 5) {
      codestream << "                         \"vprefetch0 " << (ldb * 4 * 5) + 64 << "(%%r8)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231ps " << (ldb * 4) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm30\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 3 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 7 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 2) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 6) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 4) {
      codestream << "                         \"vprefetch0 " << (ldb * 4 * 2) + 64 << "(%%r8)\\n\\t\"" << std::endl;
    } else if (k == 5) {
      codestream << "                         \"vprefetch0 " << (ldb * 4 * 6) + 64 << "(%%r8)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231ps " << (ldb * 8) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm29\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 4 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 8 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 3) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 7) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 4) {
      codestream << "                         \"vprefetch0 " << (ldb * 4 * 3) + 64 << "(%%r8)\\n\\t\"" << std::endl;
    } else if (k == 5) {
      codestream << "                         \"vprefetch0 " << (ldb * 4 * 7) + 64 << "(%%r8)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231ps " << (ldb * 12) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm28\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 16) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm27\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 20) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm26\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 24) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm25\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 28) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm24\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 32) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm23\\n\\t\"" << std::endl;
  }

  codestream << "                         \"addq $32, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << lda * 32 << ", %%r9\\n\\t\"" << std::endl;
}

void avx512knc_kernel_16x9x9_sp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, bool preC, bool blast, bool bUseMasking) {
  for (int k = 0; k < 9; k++) {
    if (alignA == true) {
      if (bUseMasking == false) {
        codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vmovaps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    } else {
      if (bUseMasking == false) {
        codestream << "                         \"vloadunpacklps " << lda * 4 * k << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * k)+64 << "(%%r9), %%zmm0\\n\\t\"" << std::endl;
      } else {
        codestream << "                         \"vloadunpacklps " << lda * 4 * k << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
        codestream << "                         \"vloadunpackhps " << (lda * 4 * k)+64 << "(%%r9), %%zmm0{{%%k1}}\\n\\t\"" << std::endl;
      }
    }

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 1 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 5 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 8 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 3) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 4) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 7) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231ps " << k * 4 << "(%%r8){{1to16}}, %%zmm0, %%zmm31\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 2 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 6 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 4) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 4) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 8) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231ps " << (ldb * 4) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm30\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 3 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 7 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << (lda * 4) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 5) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231ps " << (ldb * 8) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm29\\n\\t\"" << std::endl;

    if (k == 0) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 4 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 1) {
      codestream << "                         \"vprefetch0 " << lda * 4 * 8 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 2) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 2) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    } else if (k == 3) {
      codestream << "                         \"vprefetch1 " << (lda * 4 * 6) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }

    codestream << "                         \"vfmadd231ps " << (ldb * 12) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm28\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 16) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm27\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 20) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm26\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 24) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm25\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 28) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm24\\n\\t\"" << std::endl;
    codestream << "                         \"vfmadd231ps " << (ldb * 32) + (k * 4) << "(%%r8){{1to16}}, %%zmm0, %%zmm23\\n\\t\"" << std::endl;
  }

  codestream << "                         \"addq $" << lda * 36 << ", %%r9\\n\\t\"" << std::endl;
}
#endif

void avx512knc_generate_inner_k_loop_sp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, bool bUseMasking) {
  int k_blocking = 8;

  if (K % k_blocking == 0 ) {
    if (k_blocking == 8) {
      avx512knc_header_kloop_sp_asm(codestream, 16, k_blocking);
      avx512knc_kernel_16xNx8_sp_asm(codestream, N, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
      avx512knc_footer_kloop_sp_asm(codestream, 16, K);
    } else {
      std::cout << " !!! ERROR, MIC k-blocking !!! " << std::endl;
      exit(-1);
    }
  } else {
    // SeisSol special kernel
    if ( K == 9 ) {
      avx512knc_kernel_16xNx9_sp_asm(codestream, N, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
    } else {
      int max_blocked_K = (K/k_blocking)*k_blocking;
      if (max_blocked_K > 0 ) {
        avx512knc_header_kloop_sp_asm(codestream, 16, k_blocking);
        avx512knc_kernel_16xNx8_sp_asm(codestream, N, lda, ldb, ldc, alignA, alignC, false, false, bUseMasking);
        avx512knc_footer_kloop_notdone_sp_asm(codestream, 16, max_blocked_K);
      }
      for (int i = max_blocked_K; i < K; i++) {
        avx512knc_kernel_16xN_sp_asm(codestream, N, lda, ldb, ldc, alignA, alignC, i-max_blocked_K, bUseMasking);
      }
      // update r8 and r9
      codestream << "                         \"addq $" << lda * 4 * (K - max_blocked_K) << ", %%r9\\n\\t\"" << std::endl;
      if (max_blocked_K > 0 ) {
        codestream << "                         \"subq $" << max_blocked_K * 4 << ", %%r8\\n\\t\"" << std::endl;
      }
    }
  }  
}

void avx512knc_generate_kernel_sp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd) {
  int mDone = (M / 16) * 16;

  avx512knc_init_registers_sp_asm(codestream);
  // multiples of 16 in M
  if (mDone > 0) {
    avx512knc_header_mloop_sp_asm(codestream, 16);
    avx512knc_load_16xN_sp_asm(codestream, N, ldc, alignC, bAdd, false);
    avx512knc_generate_inner_k_loop_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, false);
    avx512knc_store_16xN_sp_asm(codestream, N, ldc, alignC, false);
    avx512knc_footer_mloop_sp_asm(codestream, 16, K, mDone, lda);
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
        break;       
    }
    codestream << "                         \"kmov %%r14d, %%k1\\n\\t\"" << std::endl;
    avx512knc_load_16xN_sp_asm(codestream, N, ldc, alignC, bAdd, true);
    // innner loop over K
    avx512knc_generate_inner_k_loop_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, true);
    avx512knc_store_16xN_sp_asm(codestream, N, ldc, alignC, true);
  }

  avx512knc_close_sp_asm(codestream, N);
}

