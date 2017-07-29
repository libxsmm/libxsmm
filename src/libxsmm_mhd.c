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
#include <libxsmm_mhd.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION const char* libxsmm_mhd_typename(libxsmm_mhd_elemtype elemtype, size_t* elemsize)
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


LIBXSMM_API_DEFINITION libxsmm_mhd_elemtype libxsmm_mhd_typeinfo(const char* elemname)
{
  libxsmm_mhd_elemtype result = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
  if (0 == strcmp("MET_CHAR", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_CHAR;
  }
  else if (0 == strcmp("MET_UCHAR", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_U8;
  }
  else if (0 == strcmp("MET_SHORT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_I16;
  }
  else if (0 == strcmp("MET_USHORT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_U16;
  }
  else if (0 == strcmp("MET_INT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_I32;
  }
  else if (0 == strcmp("MET_INT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_I32;
  }
  else if (0 == strcmp("MET_UINT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_U32;
  }
  else if (0 == strcmp("MET_LONG", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_I64;
  }
  else if (0 == strcmp("MET_ULONG", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_U64;
  }
  else if (0 == strcmp("MET_FLOAT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_F32;
  }
  else if (0 == strcmp("MET_DOUBLE", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_F64;
  }
  return result;
}

LIBXSMM_API_INLINE int internal_mhd_read(char* buffer, char split, size_t* key_end, size_t* value_begin)
{
  int result = EXIT_SUCCESS;
  char *const isplit = strchr(buffer, split);

  if (0 != isplit) {
    char* i = isplit;
    assert(0 != key_end && 0 != value_begin);
    while (buffer != i && isspace(*--i));
    *key_end = i - buffer + 1;
    i = isplit;
    while (isspace(*++i) && '\n' != *i);
    *value_begin = i - buffer;
    while ('\n' != *i && 0 != *i) ++i;
    if ('\n' == *i) *i = 0; /* fix-up */
    if (i <= (buffer + *value_begin)) {
      result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_mhd_read_header(const char* header_filename, size_t filename_max_length,
  char* filename, size_t* ndims, size_t* size, libxsmm_mhd_elemtype* type, size_t* ncomponents,
  size_t* header_size, size_t* extension_size)
{
  int result = EXIT_SUCCESS;
  char buffer[1024];
  FILE *const file = fopen(header_filename, "rb");

  if (0 != file && 0 < filename_max_length && 0 != filename && 0 != ndims && 0 < *ndims && 0 != size && 0 != type && 0 != ncomponents) {
    if (0 != extension_size) *extension_size = 0;
    if (0 != header_size) *header_size = 0;
    memset(size, 0, *ndims * sizeof(*size));
    *type = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
    *ncomponents = 1;
    *filename = 0;

    while (0 != fgets(buffer, sizeof(buffer), file) && EXIT_SUCCESS == result) {
      size_t key_end, value_begin;
      if (EXIT_SUCCESS == internal_mhd_read(buffer, '=', &key_end, &value_begin)) {
        if (0 == strncmp("NDims", buffer, key_end)
          && key_end == strlen("NDims"))
        {
          const int value = atoi(buffer + value_begin);
          if (0 < value && value <= *ndims) {
            *ndims = value;
          }
        }
        else if (0 == strncmp("ElementNumberOfChannels", buffer, key_end)
         && key_end == strlen("ElementNumberOfChannels"))
        {
          const int value = atoi(buffer + value_begin);
          if (0 < value) {
            *ncomponents = value;
          }
          else {
            result = EXIT_FAILURE;
          }
        }
        else if (0 != extension_size
          && 0 == strncmp("ExtensionDataSize", buffer, key_end)
          && key_end == strlen("ExtensionDataSize"))
        {
          const int value = atoi(buffer + value_begin);
          if (0 <= value) {
            *extension_size = value;
          }
          else {
            result = EXIT_FAILURE;
          }
        }
        else if (0 == strncmp("ElementType", buffer, key_end)
         && key_end == strlen("ElementType"))
        {
          const libxsmm_mhd_elemtype value = libxsmm_mhd_typeinfo(buffer + value_begin);
          if (LIBXSMM_MHD_ELEMTYPE_UNKNOWN != value) {
            *type = value;
          }
        }
        else if (0 == strncmp("ElementDataFile", buffer, key_end)
         && key_end == strlen("ElementDataFile"))
        {
          const char *const value = buffer + value_begin;
          if (0 == strcmp("LOCAL", value) || 0 == strcmp(header_filename, value)) {
            if (header_size) {
              *header_size = ftell(file);
              strcpy(filename, header_filename);
              /* last statement before raw data */
              break;
            }
          }
          else {
            strcpy(filename, value);
          }
        }
        else if (0 == strncmp("DimSize", buffer, key_end)
         && key_end == strlen("DimSize"))
        {
          char* value = buffer + value_begin;
          size_t n = 0;
          while (EXIT_SUCCESS == internal_mhd_read(value, ' ', &key_end, &value_begin)) {
            const int ivalue = atoi(value);
            if (0 < ivalue) {
              *size = ivalue;
            }
            else {
              result = EXIT_FAILURE;
            }
            value += key_end + 1;
            ++size;
            ++n;
          }
          if (EXIT_SUCCESS == result && 0 != *value) {
            const int ivalue = atoi(value);
            if (0 < ivalue) {
              *size = ivalue;
            }
            else {
              result = EXIT_FAILURE;
            }
            ++n;
          }
          if (*ndims < n) {
            result = EXIT_FAILURE;
          }
        }
        else if (0 == strncmp("BinaryData", buffer, key_end)
         && key_end == strlen("BinaryData"))
        {
          const char *const value = buffer + value_begin;
          if (0 == strcmp("False", value) || 0 != strcmp("True", value)) {
            result = EXIT_FAILURE;
          }
        }
        else if (0 == strncmp("CompressedData", buffer, key_end)
         && key_end == strlen("CompressedData"))
        {
          const char *const value = buffer + value_begin;
          if (0 == strcmp("True", value) || 0 != strcmp("False", value)) {
            result = EXIT_FAILURE;
          }
        }
        else if ((0 == strncmp("BinaryDataByteOrderMSB", buffer, key_end) && key_end == strlen("BinaryDataByteOrderMSB"))
              || (0 == strncmp("ElementByteOrderMSB",    buffer, key_end) && key_end == strlen("ElementByteOrderMSB")))
        {
          const char *const value = buffer + value_begin;
          if (0 == strcmp("True", value) || 0 != strcmp("False", value)) {
            result = EXIT_FAILURE;
          }
        }
      }
    }

    if (EXIT_SUCCESS == result && (0 == *filename || LIBXSMM_MHD_ELEMTYPE_UNKNOWN == *type)) {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      size_t i;
      for (i = 0; i < *ndims; ++i) {
        if (0 == size[i]) {
          result = EXIT_FAILURE;
          break;
        }
      }
    }
#if 0
    // prefix the path of the header file to actually find the data file
    if (!filename.empty() // prevent creating only a path (without filename)
                          // quick hint to pass this block
      && (!header_size || 0 == *header_size))
    {
      const std::size_t pos = header_filename.find_last_of("/\\");
      if (std::string::npos != pos) {
        // include the path separator: pos + 1
        std::string(header_filename.begin(), header_filename.begin() + pos + 1).append(filename).swap(filename);
      }
    }
#endif
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API_INLINE int internal_mhd_write(FILE* file, const void* data,
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
      result = LIBXSMM_MAX(internal_mhd_write(file, data, data_size, size, typesize, d), result);
      data = (const char*)data + sub_size;
    }
  }

  if (EXIT_SUCCESS == result) {
    const size_t nwrite = *((0 != size && 0 < *size) ? size : data_size);
    result = (nwrite == fwrite(data, typesize, nwrite, file) ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_mhd_write(const char* filename,
  const size_t* data_size, const size_t* size, size_t ndims, size_t ncomponents,
  const void* data, libxsmm_mhd_elemtype elemtype, const double* spacing,
  const char* extension_header, const void* extension, size_t extension_size)
{
  size_t elemsize = 0;
  const char *const elemtype_name = libxsmm_mhd_typename(elemtype, &elemsize);
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
    internal_mhd_write(file, data, data_size, size, ncomponents * elemsize, ndims);

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

