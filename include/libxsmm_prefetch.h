/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_PREFETCH_H
#define LIBXSMM_PREFETCH_H

#include "libxsmm.h"

#if (0 != LIBXSMM_PREFETCH)
# define LIBXSMM_PREFETCH_DECL(TYPE, ARG) , LIBXSMM_CONCATENATE2(LIBXSMM_UNUSED_, ARG) TYPE LIBXSMM_CONCATENATE2(LIBXSMM_PREFETCH_ARG_, ARG)
# define LIBXSMM_USE(ARG) LIBXSMM_CONCATENATE2(LIBXSMM_USE_, ARG)
# if 0 != ((LIBXSMM_PREFETCH) & 4) || 0 != ((LIBXSMM_PREFETCH) & 8) || 0 != ((LIBXSMM_PREFETCH) & 16)
#   define LIBXSMM_PREFETCH_ARG_pa unused_pa
#   define LIBXSMM_PREFETCH_ARGA(ARG) , 0
#   define LIBXSMM_UNUSED_pa LIBXSMM_UNUSED_ARG
#   define LIBXSMM_USE_pa LIBXSMM_UNUSED(unused_pa)
# else
#   define LIBXSMM_PREFETCH_ARG_pa pa
#   define LIBXSMM_PREFETCH_ARGA(ARG) , ARG
#   define LIBXSMM_UNUSED_pa
#   define LIBXSMM_USE_pa
# endif
# if 0 != ((LIBXSMM_PREFETCH) & 2)
#   define LIBXSMM_PREFETCH_ARG_pb unused_pb
#   define LIBXSMM_PREFETCH_ARGB(ARG) , 0
#   define LIBXSMM_UNUSED_pb LIBXSMM_UNUSED_ARG
#   define LIBXSMM_USE_pb LIBXSMM_UNUSED(unused_pb)
# else
#   define LIBXSMM_PREFETCH_ARG_pb pb
#   define LIBXSMM_PREFETCH_ARGB(ARG) , ARG
#   define LIBXSMM_UNUSED_pb
#   define LIBXSMM_USE_pb
# endif
# if 1
#   define LIBXSMM_PREFETCH_ARG_pc unused_pc
#   define LIBXSMM_PREFETCH_ARGC(ARG) , 0
#   define LIBXSMM_UNUSED_pc LIBXSMM_UNUSED_ARG
#   define LIBXSMM_USE_pc LIBXSMM_UNUSED(unused_pc)
# else
#   define LIBXSMM_PREFETCH_ARG_pc pc
#   define LIBXSMM_PREFETCH_ARGC(ARG) , ARG
#   define LIBXSMM_UNUSED_pc
#   define LIBXSMM_USE_pc
# endif
#else
# define LIBXSMM_PREFETCH_DECL(TYPE, ARG)
# define LIBXSMM_PREFETCH_ARGA(ARG)
# define LIBXSMM_PREFETCH_ARGB(ARG)
# define LIBXSMM_PREFETCH_ARGC(ARG)
# define LIBXSMM_PREFETCH_ARG_pa 0
# define LIBXSMM_PREFETCH_ARG_pb 0
# define LIBXSMM_PREFETCH_ARG_pc 0
# define LIBXSMM_USE(ARG)
#endif

#endif /*LIBXSMM_PREFETCH_H*/
