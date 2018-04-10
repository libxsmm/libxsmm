/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_SETUP_H
#define LIBXSMM_DNN_SETUP_H

#include <libxsmm_dnn.h>

LIBXSMM_API_INTERN libxsmm_dnn_err_t  libxsmm_dnn_setup_feature_map_blocks( libxsmm_dnn_layer* handle, int *noarch );
LIBXSMM_API_INTERN void libxsmm_dnn_setup_scratch( libxsmm_dnn_layer* handle );
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_generic( libxsmm_dnn_layer* handle );
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_fwd( libxsmm_dnn_layer* handle, int *noarch );
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_bwd( libxsmm_dnn_layer* handle, int *noarch );
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_setup_upd( libxsmm_dnn_layer* handle, int *noarch );

#endif /* LIBXSMM_DNN_SETUP_H */
