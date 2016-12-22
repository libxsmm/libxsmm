/******************************************************************************
** Copyright (c) 2009-2017, Intel Corporation                                **
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
#ifndef LIBXSMM_DUMP_H
#define LIBXSMM_DUMP_H

#include <libxsmm.h>


/** Denotes the element/pixel type of an image/channel. */
typedef enum libxsmm_mhd_elemtype {
  LIBXSMM_MHD_ELEMTYPE_CHAR,
  LIBXSMM_MHD_ELEMTYPE_I8,
  LIBXSMM_MHD_ELEMTYPE_U8,
  LIBXSMM_MHD_ELEMTYPE_I16,
  LIBXSMM_MHD_ELEMTYPE_U16,
  LIBXSMM_MHD_ELEMTYPE_I32,
  LIBXSMM_MHD_ELEMTYPE_U32,
  LIBXSMM_MHD_ELEMTYPE_I64,
  LIBXSMM_MHD_ELEMTYPE_U64,
  LIBXSMM_MHD_ELEMTYPE_F32,
  LIBXSMM_MHD_ELEMTYPE_F64
} libxsmm_mhd_elemtype;


/** Returns the name and size of the element type; result may be NULL/0 in case of an unknown type. */
LIBXSMM_API const char* libxsmm_meta_image_typeinfo(libxsmm_mhd_elemtype elemtype, size_t* elemsize);


/**
 * Save a file using an extended data format, which is compatible with
 * the Meta Image Format. The file is suitable for visual inspection
 * using e.g., ITK-SNAP or ParaView.
 */
LIBXSMM_API int libxsmm_meta_image_write(const char* filename,
  /** Leading dimensions (buffer extents). */
  const size_t* data_size,
  /** Image dimensions; can be NULL/0 (data_size). */
  const size_t* size,
  /** Dimensionality i.e., number of entries in "size". */
  size_t ndims,
  /** Number of pixel components. */
  size_t ncomponents,
  /** Raw data to be saved. */
  const void* data,
  /** Storage type. */
  libxsmm_mhd_elemtype elemtype,
  /** Spacing; can be NULL. */
  const double* spacing,
  /** Extension header data; can be NULL. */
  const char* extension_header,
  /** Extension data stream; can be NULL. */
  const void* extension,
  /** Extension data size; can be NULL. */
  size_t extension_size);

#endif /*LIBXSMM_DUMP_H*/
