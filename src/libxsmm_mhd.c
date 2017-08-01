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
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_MHD_MAX_LINELENGTH)
# define LIBXSMM_MHD_MAX_LINELENGTH 1024
#endif

#if !defined(LIBXSMM_MHD_MAX_ELEMSIZE)
# define LIBXSMM_MHD_MAX_ELEMSIZE 8
#endif

#define LIBXSMM_MHD_ASSIGN_VALUE(DST_TYPE, DST, SRC, SRC_ELEMTYPE, LO, HI) { const double h = (0.5 - (DST_TYPE)0.5); \
       if (LIBXSMM_MHD_ELEMTYPE_I8  == (SRC_ELEMTYPE)) { const double d = *(const signed char*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(0 <= d ? (d * (HI) / 127.0 + h) : (d * (LO) / -128.0) - h); } \
  else if (LIBXSMM_MHD_ELEMTYPE_U8  == (SRC_ELEMTYPE)) { const double d = *(const unsigned char*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(d * (HI) / 255.0 + h); } \
  else if (LIBXSMM_MHD_ELEMTYPE_I16 == (SRC_ELEMTYPE)) { const double d = *(const short*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(0 <= d ? (d * (HI) / 32767.0 + h) : (d * (LO) / -32768.0 - h)); } \
  else if (LIBXSMM_MHD_ELEMTYPE_U16 == (SRC_ELEMTYPE)) { const double d = *(const unsigned short*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(d * (HI) / 65535.0 + h); } \
  else if (LIBXSMM_MHD_ELEMTYPE_I32 == (SRC_ELEMTYPE)) { const double d = *(const int*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(0 <= d ? (d * (HI) / 2147483647.0 + h) : (d * (LO) / -2147483648.0 - h)); } \
  else if (LIBXSMM_MHD_ELEMTYPE_U32 == (SRC_ELEMTYPE)) { const double d = *(const unsigned int*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(d * (HI) / 4294967295.0 + h); } \
  else if (LIBXSMM_MHD_ELEMTYPE_I64 == (SRC_ELEMTYPE)) { const double d = (double)*(const long long*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(0 <= d ? (d * (HI) / 9223372036854775807.0 + h) : (d * (LO) / -9223372036854775808.0 - h)); } \
  else if (LIBXSMM_MHD_ELEMTYPE_U64 == (SRC_ELEMTYPE)) { const double d = (double)*(const unsigned long long*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(d * (HI) / 18446744073709551615.0 + h); } \
  else if (LIBXSMM_MHD_ELEMTYPE_F32 == (SRC_ELEMTYPE)) { const double d = *(const float*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(0 <= d ? (d * (HI) + h) : (-d * (LO) - h)); } \
  else if (LIBXSMM_MHD_ELEMTYPE_F64 == (SRC_ELEMTYPE)) { const double d = *(const double*)(SRC); \
                                                    *((DST_TYPE*)(DST)) = (DST_TYPE)(0 <= d ? (d * (HI) + h) : (-d * (LO) - h)); } \
  else { assert(0/*should not happen*/); }}


LIBXSMM_API_DEFINITION const char* libxsmm_mhd_typename(libxsmm_mhd_elemtype type, size_t* typesize)
{
  const char* elemname = 0;
  size_t size = 0;
  switch (type) {
    case LIBXSMM_MHD_ELEMTYPE_I8:   { size = 1; elemname = "MET_CHAR";    } break;
    case LIBXSMM_MHD_ELEMTYPE_U8:   { size = 1; elemname = "MET_UCHAR";   } break;
    case LIBXSMM_MHD_ELEMTYPE_I16:  { size = 2; elemname = "MET_SHORT";   } break;
    case LIBXSMM_MHD_ELEMTYPE_U16:  { size = 2; elemname = "MET_USHORT";  } break;
    case LIBXSMM_MHD_ELEMTYPE_I32:  { size = 4; elemname = "MET_INT";     } break;
    case LIBXSMM_MHD_ELEMTYPE_U32:  { size = 4; elemname = "MET_UINT";    } break;
    case LIBXSMM_MHD_ELEMTYPE_I64:  { size = 8; elemname = "MET_LONG";    } break;
    case LIBXSMM_MHD_ELEMTYPE_U64:  { size = 8; elemname = "MET_ULONG";   } break;
    case LIBXSMM_MHD_ELEMTYPE_F32:  { size = 4; elemname = "MET_FLOAT";   } break;
    case LIBXSMM_MHD_ELEMTYPE_F64:  { size = 8; elemname = "MET_DOUBLE";  } break;
    default: ;
  }
  assert(size <= LIBXSMM_MHD_MAX_ELEMSIZE);
  if (0 != typesize) *typesize = size;
  return elemname;
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

LIBXSMM_API_INLINE int internal_mhd_readline(char* buffer, char split, size_t* key_end, size_t* value_begin)
{
  int result = EXIT_SUCCESS;
  char *const isplit = strchr(buffer, split);

  if (0 != isplit) {
    char* i = isplit;
    assert(0 != key_end && 0 != value_begin);
    while (buffer != i) { --i;  if (0 == isspace((int)(*i))) break; }
    *key_end = i - buffer + 1;
    i = isplit;
    while ('\n' != *++i) if (0 == isspace((int)(*i))) break;
    *value_begin = i - buffer;
    while (0 != *i && 0 != isprint((int)(*i))) ++i;
    if (0 == isprint((int)(*i))) *i = 0; /* fix-up */
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
  char* filename, size_t* ndims, size_t* size, size_t* ncomponents, libxsmm_mhd_elemtype* type,
  size_t* header_size, size_t* extension_size)
{
  int result = EXIT_SUCCESS;
  char buffer[LIBXSMM_MHD_MAX_LINELENGTH];
  FILE *const file = fopen(header_filename, "rb");

  if (0 != file && 0 < filename_max_length && 0 != filename && 0 != ndims && 0 < *ndims && 0 != size && 0 != type && 0 != ncomponents) {
    size_t key_end, value_begin;
    if (0 != extension_size) *extension_size = 0;
    if (0 != header_size) *header_size = 0;
    memset(size, 0, *ndims * sizeof(*size));
    *type = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
    *ncomponents = 1;
    *filename = 0;

    while (0 != fgets(buffer, sizeof(buffer), file) && EXIT_SUCCESS == result &&
      EXIT_SUCCESS == internal_mhd_readline(buffer, '=', &key_end, &value_begin))
    {
      if (0 == strncmp("NDims", buffer, key_end)
        && key_end == strlen("NDims"))
      {
        const int value = atoi(buffer + value_begin);
        if (0 < value && value <= ((int)*ndims)) {
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
            const size_t len = strlen(header_filename);
            if (len < filename_max_length) {
              strncpy(filename, header_filename, len);
              *header_size = ftell(file);
              filename[len] = 0;
            }
            else {
              result = EXIT_FAILURE;
            }
            break; /* ElementDataFile is just before the raw data */
          }
        }
        else {
          const size_t len = strlen(value);
          if (len < filename_max_length) {
            strncpy(filename, value, len);
            filename[len] = 0;
          }
          else {
            result = EXIT_FAILURE;
          }
        }
      }
      else if (0 == strncmp("DimSize", buffer, key_end)
       && key_end == strlen("DimSize"))
      {
        char* value = buffer + value_begin;
        size_t *isize = size, n = 0;
        while (EXIT_SUCCESS == internal_mhd_readline(value, ' ', &key_end, &value_begin)) {
          const int ivalue = atoi(value);
          if (0 < ivalue) {
            *isize = ivalue;
          }
          else {
            result = EXIT_FAILURE;
          }
          value += key_end + 1;
          ++isize;
          ++n;
        }
        if (EXIT_SUCCESS == result && 0 != *value) {
          const int ivalue = atoi(value);
          if (0 < ivalue) {
            *isize = ivalue;
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
    /* prefix the path of the header file to make sure that the data file can be found */
    if (EXIT_SUCCESS == result && (0 == header_size || 0 == *header_size)) {
      const char* split = header_filename + strlen(header_filename) - 1;
      while (header_filename != split && 0 == strchr("/\\", *split)) --split;
      if (header_filename < split) {
        const size_t len = strlen(filename), n = split - header_filename + 1;
        if ((len+ n) <= filename_max_length) {
          size_t i;
          for (i = 1; i <= len; ++i) {
            filename[len + n - i] = filename[len - i];
          }
          for (i = 0; i < n; ++i) {
            filename[i] = header_filename[i];
          }
        }
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API int internal_mhd_element_conversion(void* /*dst*/, libxsmm_mhd_elemtype /*dst_type*/, const void* /*src*/, libxsmm_mhd_elemtype /*src_type*/);
LIBXSMM_API_DEFINITION int internal_mhd_element_conversion(void* dst, libxsmm_mhd_elemtype dst_type, const void* src, libxsmm_mhd_elemtype src_type)
{
  int result = EXIT_SUCCESS;
  switch (dst_type) {
    case LIBXSMM_MHD_ELEMTYPE_I8: {
      LIBXSMM_MHD_ASSIGN_VALUE(signed char, dst, src, src_type, -128.0, 127.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U8: {
      LIBXSMM_MHD_ASSIGN_VALUE(unsigned char, dst, src, src_type, 0.0, 255.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I16: {
      LIBXSMM_MHD_ASSIGN_VALUE(short, dst, src, src_type, -32768.0, 32767.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U16: {
      LIBXSMM_MHD_ASSIGN_VALUE(unsigned short, dst, src, src_type, 0.0, 65535.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I32: {
      LIBXSMM_MHD_ASSIGN_VALUE(int, dst, src, src_type, -2147483648.0, 2147483647.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U32: {
      LIBXSMM_MHD_ASSIGN_VALUE(unsigned int, dst, src, src_type, 0.0, 4294967295.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I64: {
      LIBXSMM_MHD_ASSIGN_VALUE(long long, dst, src, src_type, -9223372036854775808.0, 9223372036854775807.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U64: {
      LIBXSMM_MHD_ASSIGN_VALUE(unsigned long long, dst, src, src_type, 0.0, 18446744073709551615.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_F32: {
      LIBXSMM_MHD_ASSIGN_VALUE(float, dst, src, src_type, -1.0, 1.0);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_F64: {
      LIBXSMM_MHD_ASSIGN_VALUE(double, dst, src, src_type, -1.0, 1.0);
    } break;
    default: result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_mhd_element_comparison(void* dst, libxsmm_mhd_elemtype dst_type, const void* src, libxsmm_mhd_elemtype src_type)
{
  size_t typesize;
  int result;
  if (0 != libxsmm_mhd_typename(dst_type, &typesize)) {
    if (dst_type == src_type) { /* direct comparison */
      result = (0 == memcmp(src, dst, typesize) ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    else { /* conversion into source type needed */
      char element[LIBXSMM_MHD_MAX_ELEMSIZE];
      result = internal_mhd_element_conversion(element, src_type, dst, dst_type);
      if (EXIT_SUCCESS == result) {
        if (0 != libxsmm_mhd_typename(src_type, &typesize)) {
          result = (0 == memcmp(src, element, typesize) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        else {
          result = EXIT_FAILURE;
        }
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_INLINE int internal_mhd_read(FILE* file, void* data,
  const size_t* size, const size_t* pitch, size_t ndims, size_t ncomponents,
  libxsmm_mhd_elemtype type_stored, const libxsmm_mhd_elemtype* type_data,
  libxsmm_mhd_element_handler handle_element)
{
  const libxsmm_mhd_elemtype datatype = (type_data ? *type_data : type_stored);
  size_t typesize_stored, typesize_data;
  int result = EXIT_SUCCESS;

  if (0 != libxsmm_mhd_typename(type_stored, &typesize_stored) &&
      0 != libxsmm_mhd_typename(datatype, &typesize_data))
  {
    const size_t *const extent = (0 != pitch ? pitch : size);
    assert(0 != extent);

    if (1 < ndims) {
      if (size[0] <= extent[0]) {
        const size_t d = ndims - 1;

        if (EXIT_SUCCESS == result && size[d] <= extent[d]) {
          size_t sub_size = ncomponents * typesize_data * extent[0], i;

          for (i = 1; i < d; ++i) {
            if (size[i] <= extent[i]) {
              sub_size *= extent[i];
            }
            else {
              result = EXIT_FAILURE;
              break;
            }
          }
          for (i = 0; i < size[d]; ++i) {
            result = internal_mhd_read(file, data, size, pitch, d,
              ncomponents, type_stored, type_data, handle_element);
            if (EXIT_SUCCESS == result) {
              data = ((char*)data) + sub_size;
            }
            else { /* error */
              break;
            }
          }
        }
        else {
          result = EXIT_FAILURE;
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    else if (1 == ndims) {
      if (type_stored == datatype && 0 == handle_element) {
        if (extent[0] < size[0] || size[0] != fread(data, ncomponents * typesize_stored, size[0], file)) {
          result = EXIT_FAILURE;
        }
      }
      else { /* data-conversion or custom data-handler */
        const libxsmm_mhd_element_handler handler = (0 != handle_element ? handle_element : internal_mhd_element_conversion);
        char element[LIBXSMM_MHD_MAX_ELEMSIZE];
        size_t i, j;

        for (i = 0; i < size[0]; ++i) {
          for (j = 0; j < ncomponents; ++j) {
            if (1 == fread(element, typesize_stored, 1, file) &&
              EXIT_SUCCESS == handler(data, datatype, element, type_stored))
            {
              data = ((char*)data) + typesize_data;
            }
            else {
              result = EXIT_FAILURE;
              i += size[0]; /* break outer */
              break;
            }
          }
        }
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API int libxsmm_mhd_read( const char* filename,
  const size_t* size, const size_t* pitch, size_t ndims, size_t ncomponents, size_t header_size,
  libxsmm_mhd_elemtype type_stored, const libxsmm_mhd_elemtype* type_data,
  void* data, libxsmm_mhd_element_handler handle_element,
  char* extension, size_t extension_size)
{
  int result = EXIT_SUCCESS;
  FILE *const file = (0 != filename && 0 != *filename &&
    0 != size && 0 != ndims && 0 != ncomponents &&
    LIBXSMM_MHD_ELEMTYPE_UNKNOWN != type_stored &&
    (0 == type_data || LIBXSMM_MHD_ELEMTYPE_UNKNOWN != *type_data) &&
    0 != data)
    ? fopen(filename, "rb")
    : NULL;

  if (0 != file) {
    if (0 != header_size) {
      result = fseek(file, (long)header_size, SEEK_SET);
    }
    if (EXIT_SUCCESS == result) {
      result = internal_mhd_read(file, data, size, pitch, ndims,
        ncomponents, type_stored, type_data, handle_element);
    }
    if (0 != extension && 0 < extension_size) {
      if (extension_size != fread(extension, 1, extension_size, file)) {
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API_INLINE int internal_mhd_write(FILE* file, const void* data,
  const size_t* size, const size_t* pitch, size_t typesize, size_t ndims)
{
  int result = EXIT_SUCCESS;
  const size_t *const extent = (0 != pitch ? pitch : size);

  assert(0 != extent);
  if (1 < ndims) {
    if (size[0] <= extent[0]) {
      const size_t d = ndims - 1;

      if (EXIT_SUCCESS == result && size[d] <= extent[d]) {
        size_t sub_size = typesize * extent[0], i;

        for (i = 1; i < d; ++i) {
          if (size[i] <= extent[i]) {
            sub_size *= extent[i];
          }
          else {
            result = EXIT_FAILURE;
            break;
          }
        }
        for (i = 0; i < size[d]; ++i) {
          result = internal_mhd_write(file, data, size, pitch, typesize, d);
          if (EXIT_SUCCESS == result) {
            data = ((const char*)data) + sub_size;
          }
          else { /* error */
            break;
          }
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  else if (1 == ndims) {
    if (extent[0] < size[0] || size[0] != fwrite(data, typesize, size[0], file)) {
      result = EXIT_FAILURE;
    }
  }

  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_mhd_write(const char* filename,
  const size_t* size, const size_t* pitch, size_t ndims,
  size_t ncomponents, libxsmm_mhd_elemtype type, const void* data,
  const char* extension_header, const void* extension, size_t extension_size)
{
  size_t typesize = 0;
  const char *const elemname = libxsmm_mhd_typename(type, &typesize);
  FILE *const file = (0 != filename && 0 != *filename &&
      0 != size && 0 != ndims && 0 != ncomponents &&
      0 != data && 0 != elemname)
    ? fopen(filename, "wb")
    : NULL;
  int result = EXIT_SUCCESS;

  if (0 != file) {
    size_t i;
    if (0 < fprintf(file, "NDims = %u\nElementNumberOfChannels = %u\nElementByteOrderMSB = False\nDimSize =",
      (unsigned int)ndims, (unsigned int)ncomponents))
    {
      for (i = 0; i != ndims; ++i) {
        if (0 >= fprintf(file, " %u", (unsigned int)size[i])) {
          result = EXIT_FAILURE;
          break;
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && 0 < fprintf(file, "\nElementSpacing =")) {
      for (i = 0; i != ndims; ++i) {
        if (0 >= fprintf(file, " 1.0")) {
          result = EXIT_FAILURE;
          break;
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && 0 != extension_header && 0 != *extension_header) {
      if (0 >= fprintf(file, "\n%s", extension_header)) {
        result = EXIT_FAILURE;
      }
    }
    /* size of the data, which is silently appended after the regular data section */
    if (EXIT_SUCCESS == result && 0 < extension_size) {
      if (0 >= fprintf(file, "\nExtensionDataSize = %u", (unsigned int)extension_size)) {
        result = EXIT_FAILURE;
      }
    }
    /* ElementDataFile must be the last entry before writing the data */
    if (EXIT_SUCCESS == result && 0 < fprintf(file, "\nElementType = %s\nElementDataFile = LOCAL\n", elemname)) {
      result = internal_mhd_write(file, data, size, pitch, ncomponents * typesize, ndims);
    }

    /* append the extension data after the regular data section */
    if (EXIT_SUCCESS == result && 0 < extension_size) {
      if (extension_size != fwrite(extension, 1, extension_size, file)) {
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) { /* release file handle */
      if (0 != fclose(file)) {
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}

