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
#include "libxsmm_dump.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION const char* libxsmm_meta_image_typeinfo(libxsmm_mhd_elemtype elemtype, size_t* elemsize)
{
  switch (elemtype) {
    case LIBXSMM_MHD_ELEMTYPE_CHAR: if (0 != elemsize) *elemsize = 1; return "MET_CHAR";
    case LIBXSMM_MHD_ELEMTYPE_I8:   if (0 != elemsize) *elemsize = 1; return "MET_CHAR";
    case LIBXSMM_MHD_ELEMTYPE_U8:   if (0 != elemsize) *elemsize = 1; return "MET_UCHAR";
    case LIBXSMM_MHD_ELEMTYPE_I16:  if (0 != elemsize) *elemsize = 2; return "MET_SHORT";
    case LIBXSMM_MHD_ELEMTYPE_U16:  if (0 != elemsize) *elemsize = 2; return "MET_USHORT";
    case LIBXSMM_MHD_ELEMTYPE_I32:  if (0 != elemsize) *elemsize = 4; return "MET_INT";
    case LIBXSMM_MHD_ELEMTYPE_U32:  if (0 != elemsize) *elemsize = 4; return "MET_UINT";
    case LIBXSMM_MHD_ELEMTYPE_I64:  if (0 != elemsize) *elemsize = 8; return "MET_LONG";
    case LIBXSMM_MHD_ELEMTYPE_U64:  if (0 != elemsize) *elemsize = 8; return "MET_ULONG";
    case LIBXSMM_MHD_ELEMTYPE_F32:  if (0 != elemsize) *elemsize = 4; return "MET_FLOAT";
    case LIBXSMM_MHD_ELEMTYPE_F64:  if (0 != elemsize) *elemsize = 8; return "MET_DOUBLE";
    default: if (0 != elemsize) *elemsize = 0; return 0;
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE int internal_write(FILE* file, const void* data,
  const size_t* data_size, const size_t* size, size_t typesize,
  size_t ndims)
{
  int result = EXIT_SUCCESS;

  assert(0 != data_size);
  if (1 < ndims) {
    size_t d = ndims - 1, i;
    size_t sub_size = LIBXSMM_MAX(data_size[0], 0 != size ? size[0] : 0);
    for (i = 1; i < d; ++i) {
      sub_size *= LIBXSMM_MAX(data_size[i], 0 != size ? size[i] : 0);
    }
    for (i = 0; i < LIBXSMM_MAX(data_size[d], 0 != size ? size[d] : 0); ++i) {
      result = LIBXSMM_MAX(internal_write(file, data, data_size, size, typesize, d), result);
      data = (const char*)data + sub_size;
    }
  }

  if (EXIT_SUCCESS == result) {
    const size_t nwrite = *((0 != size && 0 < *size) ? size : data_size);
    result = (nwrite == fwrite(data, typesize, nwrite, file) ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_meta_image_write(const char* filename,
  const size_t* data_size, const size_t* size, size_t ndims, size_t ncomponents,
  const void* data, libxsmm_mhd_elemtype elemtype, const double* spacing,
  const char* extension_header, const void* extension, size_t extension_size)
{
  size_t elemsize = 0;
  const char *const elemtype_name = libxsmm_meta_image_typeinfo(elemtype, &elemsize);
  FILE *const file = (0 != filename && 0 != *filename &&
      0 != data_size && 0 != ndims && 0 != ncomponents &&
      0 != data && 0 != elemtype_name)
    ? fopen(filename, "wb")
    : NULL;
  int result = EXIT_SUCCESS;

  if (0 != file) {
    size_t i;

    fprintf(file, "NDims = %u\nElementNumberOfChannels = %u\nElementByteOrderMSB = False\nDimSize =",
      (unsigned int)ndims, (unsigned int)ncomponents);
    for (i = 0; i != ndims; ++i) {
      fprintf(file, " %u", (unsigned int)((0 != size && 0 < size[i]) ? size[i] : data_size[i]));
    }

    fprintf(file, "\nElementSpacing =");
    for (i = 0; i != ndims; ++i) {
      fprintf(file, " %f", (0 != spacing ? spacing[i] : 1.0));
    }

    if (0 != extension_header && 0 != *extension_header) {
      fprintf(file, "\n%s", extension_header);
    }

    /* size of the data, which is silently appended after the regular data section */
    if (0 < extension_size) {
      fprintf(file, "\nExtensionDataSize = %u", (unsigned int)extension_size);
    }

    /* ElementDataFile must be the last entry before writing the data */
    fprintf(file, "\nElementType = %s\nElementDataFile = LOCAL\n", elemtype_name);
    internal_write(file, data, data_size, size, ncomponents * elemsize, ndims);

    /* append the extension data behind the regular data section */
    if (0 < extension_size) {
      fwrite(extension, 1, extension_size, file);
    }

    /* release file handle */
    fclose(file);
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}
