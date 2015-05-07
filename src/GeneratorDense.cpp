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

/**
 * @file
 * This file is part of GemmCodeGenerator.
 *
 * @author Alexander Heinecke (alexander.heinecke AT mytum.de, http://www5.in.tum.de/wiki/index.php/Alexander_Heinecke,_M.Sc.,_M.Sc._with_honors)
 *
 * @section LICENSE
 * Copyright (c) 2012-2014, Technische Universitaet Muenchen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * <DESCRIPTION>
 */

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <assert.h>

#include "GeneratorDense.hpp"

namespace libxsmm {

  GeneratorDense::GeneratorDense() : bAlignedA_(true), bAlignedC_(true), tVec_("noarch"), bSP_(false) {
  }

  GeneratorDense::GeneratorDense(bool bAlignedA, bool bAlignedC, std::string tVec, std::string tPrefetch, bool bSP) : bAlignedA_(bAlignedA), bAlignedC_(bAlignedC), tVec_(tVec), tPrefetch_(tPrefetch), bSP_(bSP) {
  }

  GeneratorDense::~GeneratorDense() {
  }

#include "kernels_common_asm.hpp"
#include "kernels_sse3_dp_asm.hpp"
#include "kernels_sse3_sp_asm.hpp"
#include "kernels_avx1-2_common_dp_asm.hpp"
#include "kernels_avx1_dp_asm.hpp"
#include "kernels_avx2_dp_asm.hpp"
#include "kernels_avx1-2_common_sp_asm.hpp"
#include "kernels_avx1_sp_asm.hpp"
#include "kernels_avx2_sp_asm.hpp"
#include "kernels_avx512_dp_asm.hpp"
#include "kernels_avx512_sp_asm.hpp"
#include "kernels_avx512knc_dp_asm.hpp"
#include "kernels_avx512knc_sp_asm.hpp"

  std::string GeneratorDense::generate_dense(bool bIsColMajor, int M, int N, int K, int lda, int ldb, int ldc, int i_alpha, int i_beta) {
    std::stringstream codestream;
    bool alignA = false;
    bool alignC = false;

#ifdef DEBUG
    std::cout << "Generating dense matrix multiplication" << std::endl;
    std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;
    std::cout << "lda=" << lda << " ldb=" << ldb << " ldc=" << ldc << std::endl;
#endif

    assert(bIsColMajor == true);

    //////////////////////////
    //////////////////////////
    // generating SSE3 code //
    //////////////////////////
    //////////////////////////

    if ( this->tVec_.compare("wsm") == 0 ) {
      if (bSP_ == false) {
        if (lda % 2 == 0)
          alignA = true;
        else
          alignA = false;

        if (ldc % 2 == 0)
          alignC = true;
        else
          alignC = false;
      } else {
        if (lda % 4 == 0)
          alignA = true;
        else
          alignA = false;

        if (ldc % 4 == 0)
          alignC = true;
        else
          alignC = false;
      }

      // enforce external overwrite 
      alignA = alignA && this->bAlignedA_;
      alignC = alignC && this->bAlignedC_;

      codestream << "#ifdef __SSE3__" << std::endl;
      codestream << "#ifdef __AVX__" << std::endl;
      codestream << "#pragma message (\"KERNEL COMPILATION WARNING: compiling SSE3 code on AVX or newer architecture: \" __FILE__)" << std::endl;
      codestream << "#endif" << std::endl;

      //@TODO add support for alpha and beta
      if (i_alpha == -1) {
        std::cout << " !!! ERROR, SSE3, alpha not 1 !!! " << std::endl;
        exit(-1);
      }
      bool bAdd = true;
      if (i_beta == 0) {
        bAdd = false;
      }

      if (bSP_ == false) {
        sse3_generate_kernel_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, this->tPrefetch_);
      } else {
        sse3_generate_kernel_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, this->tPrefetch_);
      }

      codestream << "#else" << std::endl;
      codestream << "#pragma message (\"KERNEL COMPILATION ERROR in: \" __FILE__)" << std::endl;
      codestream << "#error No kernel was compiled, lacking support for current architecture?" << std::endl;
      codestream << "#endif" << std::endl << std::endl;
    }

    /////////////////////////
    // generating AVX code //
    /////////////////////////

    if ( this->tVec_.compare("snb") == 0 ) {
      if (bSP_ == false) {
        if (lda % 4 == 0)
          alignA = true;
        else
          alignA = false;

        if (ldc % 4 == 0)
          alignC = true;
        else
          alignC = false;
      } else {
        if (lda % 8 == 0)
          alignA = true;
        else
          alignA = false;

        if (ldc % 8 == 0)
          alignC = true;
        else
          alignC = false;
      }

      // enforce external overwrite 
      alignA = alignA && this->bAlignedA_;
      alignC = alignC && this->bAlignedC_;

      codestream << "#ifdef __AVX__" << std::endl;
      codestream << "#ifdef __AVX2__" << std::endl;
      codestream << "#pragma message (\"KERNEL COMPILATION WARNING: compiling AVX code on AVX2 or newer architecture: \" __FILE__)" << std::endl;
      codestream << "#endif" << std::endl;

      if (this->bSP_ == false) {
        avx1_generate_kernel_dp(codestream, lda, ldb, ldc, M, N, K, i_alpha, i_beta, alignA, alignC, this->tPrefetch_);
      } else {
        avx1_generate_kernel_sp(codestream, lda, ldb, ldc, M, N, K, i_alpha, i_beta, alignA, alignC, this->tPrefetch_);
      }

      codestream << "#else" << std::endl;
      codestream << "#pragma message (\"KERNEL COMPILATION ERROR in: \" __FILE__)" << std::endl;
      codestream << "#error No kernel was compiled, lacking support for current architecture?" << std::endl;
      codestream << "#endif" << std::endl << std::endl;
    }

    //////////////////////////
    // generating AVX2 code //
    //////////////////////////

    if ( this->tVec_.compare("hsw") == 0 ) {
      if (bSP_ == false) {
        if (lda % 4 == 0)
          alignA = true;
        else
          alignA = false;

        if (ldc % 4 == 0)
          alignC = true;
        else
          alignC = false;
      } else {
        if (lda % 8 == 0)
          alignA = true;
        else
          alignA = false;

        if (ldc % 8 == 0)
          alignC = true;
        else
          alignC = false;
      }

      // enforce external overwrite 
      alignA = alignA && this->bAlignedA_;
      alignC = alignC && this->bAlignedC_;

      //@TODO add support for alpha and beta
      if (i_alpha == -1) {
        std::cout << " !!! ERROR, AVX2, alpha not 1 !!! " << std::endl;
        exit(-1);
      }
      bool bAdd = true;
      if (i_beta == 0) {
        bAdd = false;
      }

      codestream << "#ifdef __AVX2__" << std::endl;
      codestream << "#ifdef __AVX512F__" << std::endl;
      codestream << "#pragma message (\"KERNEL COMPILATION WARNING: compiling AVX2 code on AVX512 or newer architecture: \" __FILE__)" << std::endl;
      codestream << "#endif" << std::endl;
      
      if (this->bSP_ == false) {
        avx2_generate_kernel_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, this->tPrefetch_);
      } else {
        avx2_generate_kernel_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, this->tPrefetch_);
      }
      
      codestream << "#else" << std::endl;
      codestream << "#pragma message (\"KERNEL COMPILATION ERROR in: \" __FILE__)" << std::endl;
      codestream << "#error No kernel was compiled, lacking support for current architecture?" << std::endl;
      codestream << "#endif" << std::endl << std::endl;
    }

    ////////////////////////////
    // generating AVX512 code //
    ////////////////////////////

    if (    (this->tVec_.compare("knc") == 0)
         || (this->tVec_.compare("knl") == 0)
         || (this->tVec_.compare("skx") == 0)
       ) {  
      if (bSP_ == false) { 
        if (lda % 8 == 0)
          alignA = true;
        else
          alignA = false;
        
        if (ldc % 8 == 0)
          alignC = true;
        else
          alignC = false;
      } else {        
        if (lda % 16 == 0)
          alignA = true;
        else
          alignA = false;
        
        if (ldc % 16 == 0)
          alignC = true;
        else
          alignC = false;
      }

      // enforce external overwrite 
      alignA = alignA && this->bAlignedA_;
      alignC = alignC && this->bAlignedC_;

      //@TODO add support for alpha and beta
      if (i_alpha == -1) {
        std::cout << " !!! ERROR, AVX512, alpha not 1 !!! " << std::endl;
        exit(-1);
      }
      bool bAdd = true;
      if (i_beta == 0) {
        bAdd = false;
      }
      
      if (this->tVec_.compare("knc") == 0) {
        codestream << "#ifdef __MIC__" << std::endl;
        if (bSP_ == false) { 
          avx512knc_generate_kernel_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd);
        } else {
          avx512knc_generate_kernel_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd);
        }
      } else if ( (this->tVec_.compare("knl") == 0) || (this->tVec_.compare("skx") == 0) ) {
        codestream << "#ifdef __AVX512F__" << std::endl;
        if (bSP_ == false) {
          avx512_generate_kernel_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, this->tPrefetch_);
        } else {
          avx512_generate_kernel_sp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, this->tPrefetch_);
        }
      } else {
        std::cout << " !!! ERROR, AVX-512 !!! " << std::endl;
        exit(-1);
      }
     
      codestream << "#else" << std::endl;
      codestream << "#pragma message (\"KERNEL COMPILATION ERROR in: \" __FILE__)" << std::endl;
      codestream << "#error No kernel was compiled, lacking support for current architecture?" << std::endl;
      codestream << "#endif" << std::endl << std::endl;
    }

    ////////////////////////////
    // generating noarch code //
    ////////////////////////////

    if (this->tVec_.compare("noarch") == 0) {
      codestream << "#pragma message (\"KERNEL COMPILATION WARNING: compiling arch-independent gemm kernel in: \" __FILE__)" << std::endl << std::endl;
      codestream << "for (unsigned int n = 0; n < " << N << "; n++) {" << std::endl; 
      if (i_beta == 0) {
        codestream << "  for(unsigned int m = 0; m < " << M << "; m++) { C[(n*" << ldc << ")+m] = 0.0; }" << std::endl << std::endl;
      }
      codestream << "  for (unsigned int k = 0; k < " << K << "; k++) {" << std::endl;
      codestream << "    #pragma simd" << std::endl;      
      codestream << "    for(unsigned int m = 0; m < " << M << "; m++) {" << std::endl;
      codestream << "      C[(n*" << ldc << ")+m] += A[(k*" << lda << ")+m] * B[(n*" << ldb << ")+k];" << std::endl;
      codestream << "    }" << std::endl;
      codestream << "  }" << std::endl;
      codestream << "}" << std::endl;
    }
   
    ////////////////////////////
    // generating flop count  //
    ////////////////////////////

    codestream << "#ifndef NDEBUG" << std::endl;
    codestream << "#ifdef _OPENMP" << std::endl;
    codestream << "#pragma omp atomic" << std::endl;
    codestream << "#endif" << std::endl;
    codestream << "libxsmm_num_total_flops += " << 2 * N* M* K << ";" << std::endl;
    codestream << "#endif" << std::endl << std::endl;

    return codestream.str();
  }
}

