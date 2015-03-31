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

void init_registers_asm(std::stringstream& codestream, std::string tPrefetch) {
  codestream << "    __asm__ __volatile__(\"movq %0, %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"movq %1, %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"movq %2, %%r10\\n\\t\"" << std::endl;
  if ( (tPrefetch.compare("BL2viaC") == 0) || (tPrefetch.compare("BL1viaC") == 0) ) {
    codestream << "                         \"movq %3, %%r12\\n\\t\"" << std::endl;
  }
  if ( (tPrefetch.compare("AL1viaC") == 0) || (tPrefetch.compare("AL2") == 0) ) {
    codestream << "                         \"movq %3, %%r11\\n\\t\"" << std::endl;
  }
  codestream << "                         \"movq $0, %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
}

void close_asm(std::stringstream& codestream, std::string tPrefetch) {
  if ( (tPrefetch.compare("BL2viaC") == 0 ) || (tPrefetch.compare("BL1viaC") == 0 ) ) {
    codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"r8\",\"r9\",\"r10\",\"r12\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");" << std::endl;
  } else if ( (tPrefetch.compare("AL1viaC") == 0) || (tPrefetch.compare("AL2") == 0) ) {
    codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"r8\",\"r9\",\"r10\",\"r11\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");" << std::endl;
  } else {
    codestream << "                        : : \"m\"(B), \"m\"(A), \"m\"(C) : \"r8\",\"r9\",\"r10\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");" << std::endl;
  }
}

void header_kloop_dp_asm(std::stringstream& codestream, int m_blocking, int k_blocking) {
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"2" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << k_blocking << ", %%r13\\n\\t\"" << std::endl;
}

void header_kloop_sp_asm(std::stringstream& codestream, int m_blocking, int k_blocking) {
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"2" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << k_blocking << ", %%r13\\n\\t\"" << std::endl;
}

void footer_kloop_dp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << K * 8 << ", %%r8\\n\\t\"" << std::endl;
}

void footer_kloop_notdone_dp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
}

void footer_kloop_sp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << K * 4 << ", %%r8\\n\\t\"" << std::endl;
}

void footer_kloop_notdone_sp_asm(std::stringstream& codestream, int m_blocking, int K) {
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
}

void header_nloop_dp_asm(std::stringstream& codestream, int n_blocking) {
  codestream << "                         \"1" << n_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << n_blocking << ", %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
}

void header_nloop_sp_asm(std::stringstream& codestream, int n_blocking) {
  codestream << "                         \"1" << n_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << n_blocking << ", %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
}

void footer_nloop_dp_asm(std::stringstream& codestream, int n_blocking, int N, int M, int lda, int ldb, int ldc, std::string tPrefetch) {
  codestream << "                         \"addq $" << ((n_blocking)*ldc * 8) - (M * 8) << ", %%r10\\n\\t\"" << std::endl;
  if (tPrefetch.compare("BL2viaC") == 0) {
    codestream << "                         \"addq $" << ((n_blocking)*ldc * 8) - (M * 8) << ", %%r12\\n\\t\"" << std::endl;
  }
  codestream << "                         \"addq $" << n_blocking* ldb * 8 << ", %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << M * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << N << ", %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"jl 1" << n_blocking << "b\\n\\t\"" << std::endl;
}

void footer_nloop_sp_asm(std::stringstream& codestream, int n_blocking, int N, int M, int lda, int ldb, int ldc, std::string tPrefetch) {
  codestream << "                         \"addq $" << ((n_blocking)*ldc * 4) - (M * 4) << ", %%r10\\n\\t\"" << std::endl;
  if (tPrefetch.compare("BL2viaC") == 0) {
    codestream << "                         \"addq $" << ((n_blocking)*ldc * 4) - (M * 4) << ", %%r12\\n\\t\"" << std::endl;
  }
  codestream << "                         \"addq $" << n_blocking* ldb * 4 << ", %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << M * 4 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << N << ", %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"jl 1" << n_blocking << "b\\n\\t\"" << std::endl;
}

void header_mloop_dp_asm(std::stringstream& codestream, int m_blocking) {
  codestream << "                         \"100" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << m_blocking << ", %%r14\\n\\t\"" << std::endl;
}

void header_mloop_sp_asm(std::stringstream& codestream, int m_blocking) {
  codestream << "                         \"100" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << m_blocking << ", %%r14\\n\\t\"" << std::endl;
}

void footer_mloop_dp_asm(std::stringstream& codestream, int m_blocking, int K, int M_done, int lda, std::string tPrefetch) {
  codestream << "                         \"addq $" << m_blocking * 8 << ", %%r10\\n\\t\"" << std::endl;
  if (tPrefetch.compare("BL2viaC") == 0) {
    codestream << "                         \"addq $" << m_blocking * 8 << ", %%r12\\n\\t\"" << std::endl;
  }
  codestream << "                         \"subq $" << (K * 8 * lda) - (m_blocking * 8) << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << M_done << ", %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"jl 100" << m_blocking << "b\\n\\t\"" << std::endl;
}

void footer_mloop_sp_asm(std::stringstream& codestream, int m_blocking, int K, int M_done, int lda, std::string tPrefetch) {
  codestream << "                         \"addq $" << m_blocking * 4 << ", %%r10\\n\\t\"" << std::endl;
  if (tPrefetch.compare("BL2viaC") == 0) {
    codestream << "                         \"addq $" << m_blocking * 4 << ", %%r12\\n\\t\"" << std::endl;
  }
  codestream << "                         \"subq $" << (K * 4 * lda) - (m_blocking * 4) << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << M_done << ", %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"jl 100" << m_blocking << "b\\n\\t\"" << std::endl;
}

