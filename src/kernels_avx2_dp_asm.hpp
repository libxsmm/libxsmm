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

void avx2_load_16xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, int i_beta, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx2_load_16xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (i_beta == 1) {
    if (alignC == true) {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovapd " <<  (l_n * ldc)      * 8 << "(%%r10), %%ymm" << 4 + (4*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovapd " << ((l_n * ldc) + 4) * 8 << "(%%r10), %%ymm" << 5 + (4*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovapd " << ((l_n * ldc) + 8) * 8 << "(%%r10), %%ymm" << 6 + (4*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovapd " << ((l_n * ldc) +12) * 8 << "(%%r10), %%ymm" << 7 + (4*l_n) << "\\n\\t\"" << std::endl;
      }
    } else {
      for (int l_n = 0; l_n < max_local_N; l_n++) {
        codestream << "                         \"vmovupd " <<  (l_n * ldc)      * 8 << "(%%r10), %%ymm" << 4 + (4*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovupd " << ((l_n * ldc) + 4) * 8 << "(%%r10), %%ymm" << 5 + (4*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovupd " << ((l_n * ldc) + 8) * 8 << "(%%r10), %%ymm" << 6 + (4*l_n) << "\\n\\t\"" << std::endl;
        codestream << "                         \"vmovupd " << ((l_n * ldc) +12) * 8 << "(%%r10), %%ymm" << 7 + (4*l_n) << "\\n\\t\"" << std::endl;
      }
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vxorpd %%ymm" << 4 + (4*l_n) << ", %%ymm" << 4 + (4*l_n) << ", %%ymm" << 4 + (4*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorpd %%ymm" << 5 + (4*l_n) << ", %%ymm" << 5 + (4*l_n) << ", %%ymm" << 5 + (4*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorpd %%ymm" << 6 + (4*l_n) << ", %%ymm" << 6 + (4*l_n) << ", %%ymm" << 6 + (4*l_n) << "\\n\\t\"" << std::endl;
      codestream << "                         \"vxorpd %%ymm" << 7 + (4*l_n) << ", %%ymm" << 7 + (4*l_n) << ", %%ymm" << 7 + (4*l_n) << "\\n\\t\"" << std::endl;
    }
  }
}

void avx2_store_16xN_dp_asm(std::stringstream& codestream, int ldc, bool alignC, int max_local_N, std::string tPrefetch) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx2_load_16xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignC == true) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovapd %%ymm" << 4 + (4*l_n) << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd %%ymm" << 5 + (4*l_n) << ", " <<  ((l_n * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd %%ymm" << 6 + (4*l_n) << ", " <<  ((l_n * ldc) + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovapd %%ymm" << 7 + (4*l_n) << ", " <<  ((l_n * ldc) +12) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vmovupd %%ymm" << 4 + (4*l_n) << ", " <<   (l_n * ldc)      * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd %%ymm" << 5 + (4*l_n) << ", " <<  ((l_n * ldc) + 4) * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd %%ymm" << 6 + (4*l_n) << ", " <<  ((l_n * ldc) + 8) * 8 << "(%%r10)\\n\\t\"" << std::endl;
      codestream << "                         \"vmovupd %%ymm" << 7 + (4*l_n) << ", " <<  ((l_n * ldc) +12) * 8 << "(%%r10)\\n\\t\"" << std::endl;
    }
  }

  if ( (tPrefetch.compare("BL2viaC") == 0) ) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"prefetcht1 " <<  (l_n * ldc)      * 8 << "(%%r12)\\n\\t\"" << std::endl;
      codestream << "                         \"prefetcht1 " << ((l_n * ldc) + 8) * 8 << "(%%r12)\\n\\t\"" << std::endl;
    }
  }
}

void avx2_kernel_16xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, int call, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx2_kernel_16xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (call != (-1)) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  for (int l_m = 0; l_m < 4; l_m++) {
    if (alignA == true) {
      codestream << "                         \"vmovapd " << 32 * l_m << "(%%r9), %%ymm3\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovupd " << 32 * l_m << "(%%r9), %%ymm3\\n\\t\"" << std::endl;
    }

    for (int l_n = 0; l_n < max_local_N; l_n++) {
      if ((l_m == 3) && (l_n == 0)) {
        codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
      }
      codestream << "                         \"vfmadd231pd %%ymm3, %%ymm" << l_n << ", %%ymm" <<  4 + l_m + (4*l_n) << "\\n\\t\"" << std::endl;
    }
  }
}

void avx2_kernel_12xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, int call, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx2_kernel_12xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (call != -1) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  for (int l_m = 0; l_m < 3; l_m++) {
    if (alignA == true) {
      codestream << "                         \"vmovapd " << 32 * l_m << "(%%r9), %%ymm" << 3 + l_m << "\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovupd " << 32 * l_m << "(%%r9), %%ymm" << 3 + l_m << "\\n\\t\"" << std::endl;
    }
  }

  for (int l_m = 0; l_m < 3; l_m++) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      if ((l_m == 2) && (l_n == 0)) {
        codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
      }
      codestream << "                         \"vfmadd231pd %%ymm" << 3 + l_m << ", %%ymm" << l_n << ", %%ymm" << 7 + l_m + (3*l_n) << "\\n\\t\"" << std::endl;
    }
  }
}

void avx2_kernel_8xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, int call, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx2_kernel_8xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (call != -1) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
  } else {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      codestream << "                         \"vbroadcastsd " << (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
  }

  for (int l_m = 0; l_m < 2; l_m++) {
    if (alignA == true) {
      codestream << "                         \"vmovapd " << 32 * l_m << "(%%r9), %%ymm" << 3 + l_m << "\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovupd " << 32 * l_m << "(%%r9), %%ymm" << 3 + l_m << "\\n\\t\"" << std::endl;
    }
  }

  for (int l_m = 0; l_m < 2; l_m++) {
    for (int l_n = 0; l_n < max_local_N; l_n++) {
      if ((l_m == 1) && (l_n == 0)) {
        codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
      }
      codestream << "                         \"vfmadd231pd %%ymm" << 3 + l_m << ", %%ymm" << l_n << ", %%ymm" << 10 + l_m +(2*l_n) << "\\n\\t\"" << std::endl;
    }
  }
}

void avx2_kernel_4xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, int call, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx2_kernel_4xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%ymm3\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    if (l_n == 0) {
      codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
    }
    if (call != -1) {
      codestream << "                         \"vbroadcastsd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vbroadcastsd " << (ldb * l_n * 8) << "(%%r8), %%ymm" << l_n << "\\n\\t\"" << std::endl;
      if (l_n == (max_local_N - 1)) {
        codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
      }
    }
    codestream << "                         \"vfmadd231pd %%ymm3, %%ymm" << l_n << ", %%ymm" << 13 + l_n << "\\n\\t\"" << std::endl;
  }
}

void avx2_kernel_2xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, int call, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx2_kernel_2xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  if (alignA == true) {
    codestream << "                         \"vmovapd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  } else {
    codestream << "                         \"vmovupd (%%r9), %%xmm3\\n\\t\"" << std::endl;
  }

  for (int l_n = 0; l_n < max_local_N; l_n++) {
    if (l_n == 0) {
      codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
    }
    if (call != -1) {
      codestream << "                         \"vmovddup " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%xmm" << l_n << "\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vmovddup " << (ldb * l_n * 8) << "(%%r8), %%xmm" << l_n << "\\n\\t\"" << std::endl;
      if (l_n == (max_local_N - 1)) {
        codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
      }
    }
    codestream << "                         \"vfmadd231pd %%xmm3, %%xmm" << l_n << ", %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
  }
}

void avx2_kernel_1xN_dp_asm(std::stringstream& codestream, int lda, int ldb, int ldc, bool alignA, bool alignC, int call, int max_local_N) {
  if ( (max_local_N > 3) || (max_local_N < 1) ) {
    std::cout << " !!! ERROR, avx2_kernel_1xN_dp_asm, N smaller 1 or larger 3!!! " << std::endl;
    exit(-1);
  }

  codestream << "                         \"vmovsd (%%r9), %%xmm3\\n\\t\"" << std::endl;
 
  for (int l_n = 0; l_n < max_local_N; l_n++) {
    if (l_n == 0) {
      codestream << "                         \"addq $" << (lda) * 8 << ", %%r9\\n\\t\"" << std::endl;
    }
    if (call != -1) {
      codestream << "                         \"vfmadd231sd " << (8 * call) + (ldb * l_n * 8) << "(%%r8), %%xmm3, %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
    } else {
      codestream << "                         \"vfmadd231sd " << (ldb * l_n * 8) << "(%%r8), %%xmm3, %%xmm" << 13 + l_n << "\\n\\t\"" << std::endl;
      if (l_n == (max_local_N - 1)) {
        codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
      }
    }
  }
}

void avx2_generate_kernel_dp(std::stringstream& codestream, int lda, int ldb, int ldc, int M, int N, int K, bool alignA, bool alignC, bool bAdd, std::string tPrefetch) {
  // functions pointers to different m_blockings
  void (*l_generatorLoad)(std::stringstream&, int, bool, int, int);
  void (*l_generatorStore)(std::stringstream&, int, bool, int, std::string);
  void (*l_generatorCompute)(std::stringstream&, int, int, int, bool, bool, int, int);

  // @TODO fix
  int i_beta = 1;
  if (bAdd == false)
    i_beta = 0;

  init_registers_asm(codestream, tPrefetch);

  int nDone = 0;
  int nDone_old = 0;
  int n_blocking = 3;

  // apply n_blocking
  while (nDone != N) {
    nDone_old = nDone;
    nDone = nDone + (((N - nDone_old) / n_blocking) * n_blocking);

    if (nDone != nDone_old && nDone > 0) {
      header_nloop_dp_asm(codestream, n_blocking);
  
      int k_blocking = 4;
      int k_threshold = 30;
      int mDone = 0;
      int mDone_old = 0;
      int m_blocking = 16;

      // apply m_blocking
      while (mDone != M) {
        if (mDone == 0) {
          mDone_old = mDone;
          if (M == 56) {
            mDone = 32;
          } else {
            mDone = mDone + (((M - mDone_old) / m_blocking) * m_blocking);
          }  
        } else {
          mDone_old = mDone;
          mDone = mDone + (((M - mDone_old) / m_blocking) * m_blocking);
        }

        // switch to a different m_blocking
        if (m_blocking == 16) {
          l_generatorLoad = &avx2_load_16xN_dp_asm;
          l_generatorStore = &avx2_store_16xN_dp_asm;
          l_generatorCompute = &avx2_kernel_16xN_dp_asm;
        } else if (m_blocking == 12) {
          l_generatorLoad = &avx_load_12xN_dp_asm;
          l_generatorStore = &avx_store_12xN_dp_asm;
          l_generatorCompute = &avx2_kernel_12xN_dp_asm;
        } else if (m_blocking == 8) {
          l_generatorLoad = &avx_load_8xN_dp_asm;
          l_generatorStore = &avx_store_8xN_dp_asm;
          l_generatorCompute = &avx2_kernel_8xN_dp_asm;
        } else if (m_blocking == 4) {
          l_generatorLoad = &avx_load_4xN_dp_asm;
          l_generatorStore = &avx_store_4xN_dp_asm;
          l_generatorCompute = &avx2_kernel_4xN_dp_asm;
        } else if (m_blocking == 2) {
          l_generatorLoad = &avx_load_2xN_dp_asm;
          l_generatorStore = &avx_store_2xN_dp_asm;
          l_generatorCompute = &avx2_kernel_2xN_dp_asm;
        } else if (m_blocking == 1) {
          l_generatorLoad = &avx_load_1xN_dp_asm;
          l_generatorStore = &avx_store_1xN_dp_asm;
          l_generatorCompute = &avx2_kernel_1xN_dp_asm;      
        } else {
          std::cout << " !!! ERROR, avx2_generate_kernel_dp, m_blocking is out of range!!! " << std::endl;
          exit(-1);
        }

        if (mDone != mDone_old && mDone > 0) {
          header_mloop_dp_asm(codestream, m_blocking);
          (*l_generatorLoad)(codestream, ldc, alignC, i_beta, n_blocking);

          if ((K % k_blocking) == 0 && K > k_threshold) {
            header_kloop_dp_asm(codestream, m_blocking, k_blocking);

            for (int k = 0; k < k_blocking; k++) {
	      (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, -1, n_blocking);
            }

            footer_kloop_dp_asm(codestream, m_blocking, K);
          } else {
            // we want to fully unroll
            if (K <= k_threshold) {
              for (int k = 0; k < K; k++) {
	        (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, k, n_blocking);
	      }
            } else {
	      // we want to block, but K % k_blocking != 0
	      int max_blocked_K = (K/k_blocking)*k_blocking;
	      if (max_blocked_K > 0 ) {
	        header_kloop_dp_asm(codestream, m_blocking, k_blocking);
	        for (int k = 0; k < k_blocking; k++) {
	          (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, -1, n_blocking);
	        }
	        footer_kloop_notdone_dp_asm(codestream, m_blocking, max_blocked_K );
	      }
	      if (max_blocked_K > 0 ) {
	        codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
	      }
	      for (int k = max_blocked_K; k < K; k++) {
	        (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, k, n_blocking);
	      }
            }
          }

          (*l_generatorStore)(codestream, ldc, alignC, n_blocking, tPrefetch);
          footer_mloop_dp_asm(codestream, m_blocking, K, mDone, lda, tPrefetch);
        }

        // switch to a different m_blocking
        if (m_blocking == 2) {
          m_blocking = 1;
        } else if (m_blocking == 4) {
          m_blocking = 2;
        } else if (m_blocking == 8) {
          m_blocking = 4;
        } else if (m_blocking == 12) {
          m_blocking = 8;
        } else if (m_blocking == 16) {
          m_blocking = 12;
        } else {
          // we are done with m_blocking
        }
      }

      footer_nloop_dp_asm(codestream, n_blocking, nDone, M, lda, ldb, ldc, tPrefetch);
    }

    // switch to a different n_blocking
    if (n_blocking == 2) {
      n_blocking = 1;
    } else if (n_blocking == 3) {
      n_blocking = 2;
    } else {
      // we are done with n_blocking
    }
  }

  close_asm(codestream, tPrefetch);
}

