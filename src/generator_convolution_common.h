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
/* Alexander Heinecke, Rajkishore Barik (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_DENSE_COMMON_H
#define GENERATOR_DENSE_COMMON_H

#include "generator_common.h"

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_m_loop( libxsmm_generated_code*                  io_generated_code,
                                                 libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                 const libxsmm_matcopy_kernel_config*      i_kernel_config,
                                                 const unsigned int                        i_gp_reg_m_loop );

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_m_loop( libxsmm_generated_code*                      io_generated_code,
                                                 libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                 const libxsmm_matcopy_kernel_config*          i_kernel_config,
                                                 const unsigned int                            i_gp_reg_m_loop,
                                                 const unsigned int                            i_m );

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_n_loop( libxsmm_generated_code*                  io_generated_code,
                                                 libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                 const libxsmm_matcopy_kernel_config*      i_kernel_config,
                                                 const unsigned int                        i_gp_reg_n_loop );

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_n_loop( libxsmm_generated_code*                      io_generated_code,
                                                 libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                 const libxsmm_matcopy_kernel_config*          i_kernel_config,
                                                 const unsigned int                            i_gp_reg_n_loop,
                                                 const unsigned int                            i_n );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_header_oi_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_oi_loop );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_footer_oi_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_oi_loop,
                                                   const unsigned int                            i_oi );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_header_oj_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_oj_loop );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_footer_oj_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_oj_loop,
                                                   const unsigned int                            i_oj );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_header_kh_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_kh_loop );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_footer_kh_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_kh_loop,
                                                   const unsigned int                            i_kh );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_header_kw_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_kw_loop );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_footer_kw_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_kw_loop,
                                                   const unsigned int                            i_kw );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_header_ifm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ifmInner_loop,
                                                    const unsigned int                        i_unrolled_trips );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_footer_ifm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ifmInner_loop,
                                                    const unsigned int                        i_trip_count );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_forward_load_output( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_forward_descriptor*     i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_backward_load_input( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_backward_descriptor*     i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_load_weight( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_weight_update_descriptor*     i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_forward_store_output( libxsmm_generated_code*                           io_generated_code,
                                                         const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                         const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                         const libxsmm_convolution_forward_descriptor*     i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_backward_store_input( libxsmm_generated_code*                           io_generated_code,
                                                         const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                         const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                         const libxsmm_convolution_backward_descriptor*     i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_store_weight( libxsmm_generated_code*                           io_generated_code,
                                                         const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                         const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                         const libxsmm_convolution_weight_update_descriptor*     i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_transpose_store_weight( libxsmm_generated_code*                           io_generated_code,
                                                         const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                         const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                         const libxsmm_convolution_weight_update_descriptor*     i_conv_desc );
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_header_ofm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ofmInner_loop,
                                                    const unsigned int                        i_unrolled_trips );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_footer_ofm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ofmInner_loop,
                                                    const unsigned int                        i_trip_count ) ;

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_backward_load_weight( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_backward_descriptor*     i_conv_desc ) ;

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_backward_fma( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_backward_descriptor*     i_conv_desc );

LIBXSMM_INTERNAL_API
void libxsmm_reset_x86_convolution_forward_gp_reg_mapping( libxsmm_convolution_forward_gp_reg_mapping* io_gp_reg_mapping );

LIBXSMM_INTERNAL_API
void libxsmm_reset_x86_convolution_backward_gp_reg_mapping( libxsmm_convolution_backward_gp_reg_mapping* io_gp_reg_mapping );

LIBXSMM_INTERNAL_API
void libxsmm_reset_x86_convolution_weight_update_gp_reg_mapping( libxsmm_convolution_weight_update_gp_reg_mapping* io_gp_reg_mapping );

LIBXSMM_INTERNAL_API
void libxsmm_generator_init_convolution_kernel_config( libxsmm_convolution_kernel_config* io_conv_kernel_config );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_store_weight( libxsmm_generated_code*                                 io_generated_code,
                                                               const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                               const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                               const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc) ;

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_load_weight( libxsmm_generated_code*                                 io_generated_code,
                                                              const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                              const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                              const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc) ;

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_transpose_load_weight( libxsmm_generated_code*                                 io_generated_code,
                                                              const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                              const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                              const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc) ;

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_header_ofw_loop( libxsmm_generated_code*                  io_generated_code,
                                                    libxsmm_loop_label_tracker*              io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config* i_conv_kernel_config,
                                                    const unsigned int                       i_gp_reg_oi_loop );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_footer_ofw_loop( libxsmm_generated_code*                  io_generated_code,
                                                    libxsmm_loop_label_tracker*              io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config* i_conv_kernel_config,
                                                    const unsigned int                       i_gp_reg_oi_loop, const unsigned int                            i_ofw );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_header_ofh_loop( libxsmm_generated_code*                  io_generated_code,
                                                    libxsmm_loop_label_tracker*              io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config* i_conv_kernel_config,
                                                    const unsigned int                       i_gp_reg_ofh_loop );

LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_footer_ofh_loop( libxsmm_generated_code*                  io_generated_code,
                                                    libxsmm_loop_label_tracker*              io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config* i_conv_kernel_config,
                                                    const unsigned int                       i_gp_reg_ofh_loop,
                                                    const unsigned int                       i_ofh );

#endif /* GENERATOR_DENSE_COMMON_H */

