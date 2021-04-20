/**
 * @file This file is part of EDGE.
 *
 * @author Alexander Breuer (anbreuer AT ucsd.edu)
 *         Alexander Heinecke (alexander.heinecke AT intel.com)
 *
 * @section LICENSE
 * Copyright (c) 2016-2017, Regents of the University of California
 * Copyright (c) 2016, Intel Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * Data structures of the fused LIBXSMM, matrix-matrix multiplication kernels.
 **/
#ifndef EDGE_DATA_MM_XSMM_FUSED_HPP
#define EDGE_DATA_MM_XSMM_FUSED_HPP

#include <vector>
#include "constants.hpp"
#ifdef PP_USE_EDGE_IO
#include "io/logging.h"
#endif

#include <libxsmm.h>

namespace edge {
  namespace data {
    template< typename TL_T_REAL >
    class MmXsmmFused;

    template<>
    class MmXsmmFused< float >;

    template<>
    class MmXsmmFused< double >;
  }
}

/**
 * Holds LIBXSMM kernels for non-fused, single precision simulations.
 **/
template<>
class edge::data::MmXsmmFused< float > {
  private:
    //! gemm descriptors of libxsmm
    std::vector< const libxsmm_gemm_descriptor* > m_descs;

  public:
    //! generated kernels of libxsmm
    std::vector< libxsmm_smmfunction > m_kernels;

    /**
     * Adds a sparse libxsmm-kernel for the given matrix in CSR- or CSC-format.
     *
     * @param i_csr true for CSR, false for CSC.
     * @param i_ptr row-pointer of CSR or column pointer of CSC; last element holds the number of non-zero entries.
     * @param i_idx column index of CSR or row index of CSC.
     * @param i_m number of rows in column-major A and C.
     * @param i_n number of columns in column-major B and C.
     * @param i_k number of columns/rows in column-major A/B.
     * @param i_ldA leading dimension of column-major A.
     * @param i_ldB leading dimension of column-major B.
     * @param i_ldC leading dimension of column-major C.
     * @param i_alpha alpha parameter (needs to be 1.0 for now).
     * @param i_beta beta parameter (need to be 0.0/1.0 for now).
     * @param i_prefetch prefetch strategy.
     **/
    void add( bool                              i_csr,
              unsigned int               const *i_ptr,
              unsigned int               const *i_idx,
              float                      const *i_val,
              unsigned int                      i_m,
              unsigned int                      i_n,
              unsigned int                      i_k,
              unsigned int                      i_ldA,
              unsigned int                      i_ldB,
              unsigned int                      i_ldC,
              float                             i_alpha,
              float                             i_beta,
              libxsmm_gemm_prefetch_type        i_prefetch ) {
#ifdef PP_USE_EDGE_IO
      EDGE_VLOG(1) << "  adding single precision XSMM-kernel #" << m_kernels.size() << " (sparse)"
                   << " M=" << i_m << " N=" << i_n << " K=" << i_k
                   << " ldA=" << i_ldA << " ldB=" << i_ldB << " ldC=" << i_ldC
                   << " alpha=" << i_alpha << " beta=" << i_beta;
#elif defined PP_REPRODUCER_DEBUG
      std::cout << "  adding single precision XSMM-kernel #" << m_kernels.size() << " (sparse)"
                << " M=" << i_m << " N=" << i_n << " K=" << i_k
                << " ldA=" << i_ldA << " ldB=" << i_ldB << " ldC=" << i_ldC
                << " alpha=" << i_alpha << " beta=" << i_beta << std::endl;
#endif

#ifdef PP_USE_EDGE_IO
#if defined(__AVX512F__)
      EDGE_CHECK( PP_N_CRUNS == 16 );
#elif defined(__AVX2__)
      EDGE_CHECK( PP_N_CRUNS == 8 );
#endif
#endif

      // add description
      libxsmm_descriptor_blob l_xgemmBlob;
      const libxsmm_gemm_descriptor* l_desc = 0;
      const int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
      l_desc = libxsmm_gemm_descriptor_dinit(&l_xgemmBlob, LIBXSMM_GEMM_PRECISION_F32,
        i_m, i_n, i_k, i_ldA, i_ldB, i_ldC, i_alpha, i_beta, l_flags, i_prefetch);

      m_descs.push_back( l_desc );

      // generate and store function for this kernels
      if( i_csr )
        m_kernels.push_back( libxsmm_create_packed_spxgemm_csr( m_descs.back(), PP_N_CRUNS, i_ptr, i_idx, i_val ).smm );
      else
        m_kernels.push_back( libxsmm_create_packed_spxgemm_csc( m_descs.back(), PP_N_CRUNS, i_ptr, i_idx, i_val ).smm );

#ifdef PP_USE_EDGE_IO
      // check that we generated a kernel
      EDGE_CHECK( m_kernels.back() != 0 );
#endif
    }
};

/**
 * Holds LIBXSMM kernels for non-fused, double precision simulations.
 **/
template<>
class edge::data::MmXsmmFused< double > {
  private:
    //! gemm descriptors of libxsmm
    std::vector< const libxsmm_gemm_descriptor* > m_descs;

  public:
    //! generated kernels of libxsmm
    std::vector< libxsmm_dmmfunction > m_kernels;

    /**
     * Adds a sparse libxsmm-kernel for the given matrix in CSR- or CSC-format.
     *
     * @param i_csr true for CSR, false for CSC.
     * @param i_ptr row-pointer of CSR or column pointer of CSC; last element holds the number of non-zero entries.
     * @param i_idx column index of CSR or row index of CSC.
     * @param i_val non-zero values.
     * @param i_m number of rows in column-major A and C.
     * @param i_n number of columns in column-major B and C.
     * @param i_k number of columns/rows in column-major A/B.
     * @param i_ldA leading dimension of column-major A.
     * @param i_ldB leading dimension of column-major B.
     * @param i_ldC leading dimension of column-major C.
     * @param i_alpha alpha parameter (needs to be 1.0 for now).
     * @param i_beta beta parameter (need to be 0.0/1.0 for now).
     * @param i_prefetch prefetch strategy.
     **/
    void add( bool                              i_csr,
              unsigned int               const *i_ptr,
              unsigned int               const *i_idx,
              double                     const *i_val,
              unsigned int                      i_m,
              unsigned int                      i_n,
              unsigned int                      i_k,
              unsigned int                      i_ldA,
              unsigned int                      i_ldB,
              unsigned int                      i_ldC,
              double                            i_alpha,
              double                            i_beta,
              libxsmm_gemm_prefetch_type        i_prefetch ) {
#ifdef PP_USE_EDGE_IO
      EDGE_VLOG(1) << "  adding double precision XSMM-kernel #" << m_kernels.size() << " (sparse)"
                   << " M=" << i_m << " N=" << i_n << " K=" << i_k
                   << " ldA=" << i_ldA << " ldB=" << i_ldB << " ldC=" << i_ldC
                   << " alpha=" << i_alpha << " beta=" << i_beta;
#elif defined PP_REPRODUCER_DEBUG
      std::cout << "  adding double precision XSMM-kernel #" << m_kernels.size() << " (sparse)"
                << " M=" << i_m << " N=" << i_n << " K=" << i_k
                << " ldA=" << i_ldA << " ldB=" << i_ldB << " ldC=" << i_ldC
                << " alpha=" << i_alpha << " beta=" << i_beta << std::endl;
#endif

#ifdef PP_USE_EDGE_IO
      // check fused runs
#if defined(__AVX512F__)
      EDGE_CHECK( PP_N_CRUNS == 8 );
#elif defined(__AVX2__)
      EDGE_CHECK( PP_N_CRUNS == 4 );
#endif
#endif

      // add description
      libxsmm_descriptor_blob l_xgemmBlob;
      const libxsmm_gemm_descriptor* l_desc = 0;
      const int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
      l_desc = libxsmm_gemm_descriptor_dinit(&l_xgemmBlob, LIBXSMM_GEMM_PRECISION_F64,
        i_m, i_n, i_k, i_ldA, i_ldB, i_ldC, i_alpha, i_beta, l_flags, i_prefetch);

      m_descs.push_back( l_desc );

      // generate and store function for this kernels
      if( i_csr )
        m_kernels.push_back( libxsmm_create_xcsr_soa( m_descs.back(), i_ptr, i_idx, i_val ).dmm );
      else
        m_kernels.push_back( libxsmm_create_xcsc_soa( m_descs.back(), i_ptr, i_idx, i_val ).dmm );

#ifdef PP_USE_EDGE_IO
      // check that we generated a kernel
      EDGE_CHECK( m_kernels.back() != 0 );
#endif
    }
};
#endif

