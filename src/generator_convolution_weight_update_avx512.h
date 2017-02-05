/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
/* Rajkishore Barik (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_CONVOLUTION_WEIGHT_UPDATE_AVX512_H
#define GENERATOR_CONVOLUTION_WEIGHT_UPDATE_AVX512_H

#include "generator_common.h"
#include "generator_convolution_common.h"

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_avx512_kernel( libxsmm_generated_code*     io_generated_code,
                                                          const libxsmm_convolution_weight_update_descriptor* i_conv_desc,
                                                          const char*                       i_arch );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_avx512_init_weight_strides( libxsmm_generated_code*                          io_generated_code,
                                                                      const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                      const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                                      const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_avx512_ofwloop( libxsmm_generated_code*                           io_generated_code,
                                                           const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                           const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                           const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
                                                           const unsigned int                                      i_ofh_unroll,
            const unsigned int ofh_trip_counter,
                                                           const int                                               no_unroll_no_block);

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_avx512_ofwloop_sfma( libxsmm_generated_code*                           io_generated_code,
                                                                const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                                const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
                                                                const unsigned int                                      i_ofh_unroll,
            const unsigned int ofh_trip_counter,
                                                                const int                                               no_unroll_no_block);

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_transpose_avx512_init_weight_strides( libxsmm_generated_code*                          io_generated_code,
                                                                      const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                      const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                                      const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop( libxsmm_generated_code*                           io_generated_code,
                                                           const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                           const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                           const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
                                                           const unsigned int                                      i_ofh_unroll,
            const unsigned int ofh_trip_counter,
                                                           const int                                               no_unroll_no_block);

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop_sfma( libxsmm_generated_code*                           io_generated_code,
                                                                const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                                const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
                                                                const unsigned int                                      i_ofh_unroll,
            const unsigned int ofh_trip_counter,
                                                                const int                                               no_unroll_no_block);

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop_qfma( libxsmm_generated_code*                           io_generated_code,
                                                                const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                                const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
                                                                const unsigned int                                      i_ofh_unroll,
            const unsigned int ofh_trip_counter,
                                                                const int                                               no_unroll_no_block);

#endif /* GENERATOR_CONVOLUTION_WEIGHT_UPDATE_AVX512_H */

