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
#include <immintrin.h>

#include "GeneratorCSC.hpp"
#include "ReaderCSC.hpp"

#define C_UNROLLING_FACTOR_RIGHT 1
//#define USE_THREE_ELEMENT_AVX_VECTORIZATION

namespace seissolgen {

  GeneratorCSC::GeneratorCSC() : bAdd_(true), bSP_(false), tVec_("noarch") {
  }

  GeneratorCSC::GeneratorCSC(bool bAdd, bool bSP, std::string tVec) : bAdd_(bAdd), bSP_(bSP), tVec_(tVec) {
  }

  GeneratorCSC::~GeneratorCSC() {
  }

  void GeneratorCSC::generate_code_left_innerloop_scalar(std::stringstream& codestream, int ldc, int l, int z, int* rowidx, int* colidx) {
    if (this->bSP_ == true) {
      codestream << "__m128 c" << l << "_" << z << " = _mm_load_ss(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "]);" << std::endl;
      codestream << "__m128 a" << l << "_" << z << " = _mm_load_ss(&values[" << colidx[l] + z << "]);" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm_add_ss(c" << l << "_" << z << ", _mm_mul_ss(a" << l << "_" << z << ", b" << l << "));" << std::endl;
      codestream << "_mm_store_ss(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "], c" << l << "_" << z << ");" << std::endl;
    } else {
      codestream << "__m128d c" << l << "_" << z << " = _mm_load_sd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "]);" << std::endl;
      codestream << "__m128d a" << l << "_" << z << " = _mm_load_sd(&values[" << colidx[l] + z << "]);" << std::endl;
      codestream << "#if defined(__SSE3__) && defined(__AVX__)" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm_add_sd(c" << l << "_" << z << ", _mm_mul_sd(a" << l << "_" << z << ", _mm256_castpd256_pd128(b" << l << ")));" << std::endl;
      codestream << "#endif" << std::endl;
      codestream << "#if defined(__SSE3__) && !defined(__AVX__)" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm_add_sd(c" << l << "_" << z << ", _mm_mul_sd(a" << l << "_" << z << ", b" << l << "));" << std::endl;
      codestream << "#endif" << std::endl;
      codestream << "_mm_store_sd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "], c" << l << "_" << z << ");" << std::endl;
    }
  }

  void GeneratorCSC::generate_code_left_innerloop_2vector(std::stringstream& codestream, int ldc, int l, int z, int* rowidx, int* colidx) {
    if (this->bSP_ == true) {
      codestream << "__m128 c" << l << "_" << z << " = _mm_castpd_ps(_mm_load_sd((double*)(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "])));" << std::endl;
      codestream << "__m128 a" << l << "_" << z << " = _mm_castpd_ps(_mm_load_sd((double*)(&values[" << colidx[l] + z << "])));" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm_add_ps(c" << l << "_" << z << ", _mm_mul_ps(a" << l << "_" << z << ", b" << l << "));" << std::endl;
      codestream << "_mm_store_sd((double*)(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "]), _mm_castps_pd(c" << l << "_" << z << "));" << std::endl;
    } else {
      codestream << "__m128d c" << l << "_" << z << " = _mm_loadu_pd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "]);" << std::endl;
      codestream << "__m128d a" << l << "_" << z << " = _mm_loadu_pd(&values[" << colidx[l] + z << "]);" << std::endl;
      codestream << "#if defined(__SSE3__) && defined(__AVX__)" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm_add_pd(c" << l << "_" << z << ", _mm_mul_pd(a" << l << "_" << z << ", _mm256_castpd256_pd128(b" << l << ")));" << std::endl;
      codestream << "#endif" << std::endl;
      codestream << "#if defined(__SSE3__) && !defined(__AVX__)" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm_add_pd(c" << l << "_" << z << ", _mm_mul_pd(a" << l << "_" << z << ", b" << l << "));" << std::endl;
      codestream << "#endif" << std::endl;
      codestream << "_mm_storeu_pd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "], c" << l << "_" << z << ");" << std::endl;
    }
  }

  void GeneratorCSC::generate_code_left_innerloop_4vector(std::stringstream& codestream, int ldc, int l, int z, int* rowidx, int* colidx) {
    if (this->bSP_ == true) {
      codestream << "__m128 c" << l << "_" << z << " = _mm_loadu_ps(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "]);" << std::endl;
      codestream << "__m128 a" << l << "_" << z << " = _mm_loadu_ps(&values[" << colidx[l] + z << "]);" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm_add_ps(c" << l << "_" << z << ", _mm_mul_ps(a" << l << "_" << z << ", b" << l << "));" << std::endl;
      codestream << "_mm_storeu_ps(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "], c" << l << "_" << z << ");" << std::endl;
    } else {
      codestream << "#if defined(__SSE3__) && defined(__AVX__)" << std::endl;
      codestream << "__m256d c" << l << "_" << z << " = _mm256_loadu_pd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "]);" << std::endl;
      codestream << "__m256d a" << l << "_" << z << " = _mm256_loadu_pd(&values[" << colidx[l] + z << "]);" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm256_add_pd(c" << l << "_" << z << ", _mm256_mul_pd(a" << l << "_" << z << ", b" << l << "));" << std::endl;
      codestream << "_mm256_storeu_pd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "], c" << l << "_" << z << ");" << std::endl;
      codestream << "#endif" << std::endl;
      codestream << "#if defined(__SSE3__) && !defined(__AVX__)" << std::endl;
      codestream << "__m128d c" << l << "_" << z << " = _mm_loadu_pd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "]);" << std::endl;
      codestream << "__m128d a" << l << "_" << z << " = _mm_loadu_pd(&values[" << colidx[l] + z << "]);" << std::endl;
      codestream << "c" << l << "_" << z << " = _mm_add_pd(c" << l << "_" << z << ", _mm_mul_pd(a" << l << "_" << z << ", b" << l << "));" << std::endl;
      codestream << "_mm_storeu_pd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "], c" << l << "_" << z << ");" << std::endl;
      codestream << "__m128d c" << l << "_" << z + 2 << " = _mm_loadu_pd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z + 2] << "]);" << std::endl;
      codestream << "__m128d a" << l << "_" << z + 2 << " = _mm_loadu_pd(&values[" << colidx[l] + z + 2 << "]);" << std::endl;
      codestream << "c" << l << "_" << z + 2 << " = _mm_add_pd(c" << l << "_" << z + 2 << ", _mm_mul_pd(a" << l << "_" << z + 2 << ", b" << l << "));" << std::endl;
      codestream << "_mm_storeu_pd(&C[(i*" << ldc << ")+" << rowidx[colidx[l] + z + 2] << "], c" << l << "_" << z + 2 << ");" << std::endl;
      codestream << "#endif" << std::endl;
    }
  }

  std::string GeneratorCSC::generate_code_right(std::string tFilename, int nM, int nN, int nK, int lda, int ldc) {
    int* ptr_rowidx = NULL;
    int* ptr_colidx = NULL;
    double* ptr_values = NULL;
    int numRows = 0;
    int numCols = 0;
    int numElems = 0;
    std::stringstream codestream;

    ReaderCSC myReader;
    myReader.parse_file(tFilename, ptr_rowidx, ptr_colidx, ptr_values, numRows, numCols, numElems);

    int* rowidx = ptr_rowidx;
    int* colidx = ptr_colidx;
    size_t num_local_flops = 0;

#ifdef DEBUG
    std::cout << numRows << " " << numCols << " " << numElems << std::endl;
    std::cout << ptr_rowidx << " " << ptr_colidx << " " << ptr_values << std::endl;
#endif

    if (bAdd_ == false) {
      codestream << "for (int n = 0; n < " << nN << "; n++) {" << std::endl;
      if (nM > 1) {
        codestream << "  #pragma simd" << std::endl;
        codestream << "  #pragma vector aligned" << std::endl;
      }
      codestream << "  for (unsigned int m = 0; m < " << nM << "; m++) { C[(n*" << ldc << ")+m] = 0.0; }" << std::endl;
      codestream << "}" << std::endl;
    }

    if (   (tVec_.compare("noarch") == 0)
        || (tVec_.compare("wsm") == 0)
        || (tVec_.compare("snb") == 0)
        || (tVec_.compare("hsw") == 0) ) {
      if (nM > 1) {
        codestream << "#pragma simd vectorlength(2,4,8)" << std::endl;
        codestream << "#pragma vector aligned" << std::endl;   
      }
    }

    if (   (tVec_.compare("knc") == 0)
       ) {
      if (nM > 1) {
        codestream << "#pragma simd vectorlength(32)" << std::endl;
        codestream << "#pragma vector aligned" << std::endl;   
      }
    }

    codestream << "for (unsigned int i = 0; i < " << nM << "; i += 1)" << std::endl;
    codestream << "{" << std::endl;

    // generate code
    for (int t = 0; t < nN; t++) {
      int lcl_colElems = colidx[t + 1] - colidx[t];

      for (int z = 0; z < lcl_colElems; z++) {
        // loop over all cols in A
        if (rowidx[colidx[t] + z] < nK) {
          codestream << "C[(i)+(" << t* ldc << ")] += A[(i)+(" << (rowidx[colidx[t] + z]*lda) << ")] * ";
          codestream << "values[" << colidx[t] + z << "];" << std::endl;
          num_local_flops += 2;
        }
      }  
    }
    codestream << "}" << std::endl;

    codestream << "#ifndef NDEBUG" << std::endl;
    codestream << "#ifdef _OPENMP" << std::endl;
    codestream << "#pragma omp atomic" << std::endl;
    codestream << "#endif" << std::endl;
    codestream << "libxsmm_num_total_flops += " << num_local_flops*nM << ";" << std::endl;
    codestream << "#endif" << std::endl << std::endl;

    _mm_free(ptr_rowidx);
    _mm_free(ptr_colidx);
    _mm_free(ptr_values);

    return codestream.str();
  }

  std::string GeneratorCSC::generate_code_left(std::string tFilename, int nM, int nN, int nK, int ldb, int ldc) {
    int* ptr_rowidx = NULL;
    int* ptr_colidx = NULL;
    double* ptr_values = NULL;
    int numRows = 0;
    int numCols = 0;
    int numElems = 0;
    std::stringstream codestream;

    ReaderCSC myReader;
    myReader.parse_file(tFilename, ptr_rowidx, ptr_colidx, ptr_values, numRows, numCols, numElems);

    int* rowidx = ptr_rowidx;
    int* colidx = ptr_colidx;
    size_t num_local_flops = 0;

#ifdef DEBUG
    std::cout << numRows << " " << numCols << " " << numElems << std::endl;
    std::cout << ptr_rowidx << " " << ptr_colidx << " " << ptr_values << std::endl;
#endif

    // loop over the columns in C in the generated code
    codestream << "#pragma nounroll_and_jam" << std::endl;
    codestream << "for (unsigned int i = 0; i < " << nN << "; i++)" << std::endl;
    codestream << "{" << std::endl;

    // loop over columns in A, rows in B
    for (int l = 0; l < nK; l++) {

      if (bAdd_ == false) {
        // set C column to zero
        if (l == 0) {
          if (nM > 1) {
            codestream << "  #pragma simd" << std::endl;
          }
          codestream << "  for (unsigned int m = 0; m < " << nM << "; m++) {" << std::endl;
          codestream << "    C[(i*" << ldc << ")+m] = 0.0;" << std::endl;
          codestream << "  }" << std::endl;
        }
      }

      int lcl_colElems = colidx[l + 1] - colidx[l];
      codestream << "#if defined(__SSE3__) || defined(__AVX__)" << std::endl;

      if (lcl_colElems > 0) {
        if (this->bSP_ == true) {
          codestream << "#if defined(__SSE3__) && defined(__AVX__)" << std::endl;
          codestream << "__m128 b" << l << " = _mm_broadcast_ss(&B[(i*" << ldb << ")+" << l << "]);" << std::endl;
          codestream << "#endif" << std::endl;
          codestream << "#if defined(__SSE3__) && !defined(__AVX__)" << std::endl;
          codestream << "__m128 b" << l << " = _mm_load_ss(&B[(i*" << ldb << ")+" << l << "]);" << std::endl;
          codestream << "b" << l << " = _mm_shuffle_ps(b" << l << ", b" << l << ", 0x00);" << std::endl;
          codestream << "#endif" << std::endl;
        } else {
          codestream << "#if defined(__SSE3__) && defined(__AVX__)" << std::endl;
          codestream << "__m256d b" << l << " = _mm256_broadcast_sd(&B[(i*" << ldb << ")+" << l << "]);" << std::endl;
          codestream << "#endif" << std::endl;
          codestream << "#if defined(__SSE3__) && !defined(__AVX__)" << std::endl;
          codestream << "__m128d b" << l << " = _mm_loaddup_pd(&B[(i*" << ldb << ")+" << l << "]);" << std::endl;
          codestream << "#endif" << std::endl;
        }
      }

      // loop over all rows in A
      int z;
#ifdef __ICC
#pragma novector
#endif

      for (z = 0; z < lcl_colElems; z++) {
        // 4 element vector might be possible
        if ( z < (lcl_colElems - 3) ) {
          // generate 256bit vector instruction
          if ((rowidx[colidx[l] + z] + 1 == rowidx[colidx[l] + (z + 1)]) &&
              (rowidx[colidx[l] + z] + 2 == rowidx[colidx[l] + (z + 2)]) &&
              (rowidx[colidx[l] + z] + 3 == rowidx[colidx[l] + (z + 3)]) && 
              (rowidx[colidx[l] + (z + 3)] < nM)) {
            generate_code_left_innerloop_4vector(codestream, ldc, l, z, rowidx, colidx);
            //num_local_flops += 8;
            z += 3;
          }
          // generate 128bit vector instruction
          else if ((rowidx[colidx[l] + z] + 1 == rowidx[colidx[l] + (z + 1)]) &&
                   (rowidx[colidx[l] + (z + 1)] < nM) ) {
            generate_code_left_innerloop_2vector(codestream, ldc, l, z, rowidx, colidx);
            //num_local_flops += 4;
            z++;
          }
          //scalar
          else {
            if (rowidx[colidx[l] + z] < nM) {
              generate_code_left_innerloop_scalar(codestream, ldc, l, z, rowidx, colidx);
              //num_local_flops += 2;
            }
          }
        }
        // 2 element vector might be possible
        else if (z < (lcl_colElems - 1)) {
          // generate 128bit vector instruction
          if ( (rowidx[colidx[l] + z] + 1 == rowidx[colidx[l] + (z + 1)]) &&
               (rowidx[colidx[l] + (z + 1)] < nM)) {
            generate_code_left_innerloop_2vector(codestream, ldc, l, z, rowidx, colidx);
            //num_local_flops += 4;
            z++;
          }
          //scalar
          else {
            if (rowidx[colidx[l] + z] < nM) {
              generate_code_left_innerloop_scalar(codestream, ldc, l, z, rowidx, colidx);
              //num_local_flops += 2;
            }
          }
        }

        // scalar anyway
        else {
          if (rowidx[colidx[l] + z] < nM) {
            generate_code_left_innerloop_scalar(codestream, ldc, l, z, rowidx, colidx);
            //num_local_flops += 2;
          }
        }
      }

      codestream << "#else" << std::endl;

      for (z = 0; z < lcl_colElems; z++) {
        if (rowidx[colidx[l] + z] < nM) {
          codestream << "C[(i*" << ldc << ")+" << rowidx[colidx[l] + z] << "] += ";
          codestream << "values[" << colidx[l] + z << "]";
          codestream << " * B[(i*" << ldb << ")+" << l << "];" << std::endl;
          num_local_flops += 2;
        }
      }

      codestream << "#endif" << std::endl;
    }

    codestream << std::endl;
    codestream << "}" << std::endl << std::endl;

    codestream << "#ifndef NDEBUG" << std::endl;
    codestream << "#ifdef _OPENMP" << std::endl;
    codestream << "#pragma omp atomic" << std::endl;
    codestream << "#endif" << std::endl;
    codestream << "libxsmm_num_total_flops += " << num_local_flops*nN << ";" << std::endl;
    codestream << "#endif" << std::endl << std::endl;

    _mm_free(ptr_rowidx);
    _mm_free(ptr_colidx);
    _mm_free(ptr_values);

    return codestream.str();
  }

}

