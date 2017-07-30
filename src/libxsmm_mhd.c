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

#if !defined(LIBXSMM_MHD_MAX_LINELENGTH)
# define LIBXSMM_MHD_MAX_LINELENGTH 1024
#endif


LIBXSMM_API_DEFINITION const char* libxsmm_mhd_typename(libxsmm_mhd_elemtype type, size_t* typesize)
{
  switch (type) {
    case LIBXSMM_MHD_ELEMTYPE_CHAR: if (0 != typesize) *typesize = 1; return "MET_CHAR";
    case LIBXSMM_MHD_ELEMTYPE_I8:   if (0 != typesize) *typesize = 1; return "MET_CHAR";
    case LIBXSMM_MHD_ELEMTYPE_U8:   if (0 != typesize) *typesize = 1; return "MET_UCHAR";
    case LIBXSMM_MHD_ELEMTYPE_I16:  if (0 != typesize) *typesize = 2; return "MET_SHORT";
    case LIBXSMM_MHD_ELEMTYPE_U16:  if (0 != typesize) *typesize = 2; return "MET_USHORT";
    case LIBXSMM_MHD_ELEMTYPE_I32:  if (0 != typesize) *typesize = 4; return "MET_INT";
    case LIBXSMM_MHD_ELEMTYPE_U32:  if (0 != typesize) *typesize = 4; return "MET_UINT";
    case LIBXSMM_MHD_ELEMTYPE_I64:  if (0 != typesize) *typesize = 8; return "MET_LONG";
    case LIBXSMM_MHD_ELEMTYPE_U64:  if (0 != typesize) *typesize = 8; return "MET_ULONG";
    case LIBXSMM_MHD_ELEMTYPE_F32:  if (0 != typesize) *typesize = 4; return "MET_FLOAT";
    case LIBXSMM_MHD_ELEMTYPE_F64:  if (0 != typesize) *typesize = 8; return "MET_DOUBLE";
    default: if (0 != typesize) *typesize = 0; return 0;
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

LIBXSMM_API_INLINE int internal_mhd_readline(char* buffer, char split, size_t* key_end, size_t* value_begin)
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
  char* filename, size_t* ndims, size_t* size, size_t* ncomponents, libxsmm_mhd_elemtype* type,
  size_t* header_size, size_t* extension_size)
{
  int result = EXIT_SUCCESS;
  char buffer[LIBXSMM_MHD_MAX_LINELENGTH];
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
      if (EXIT_SUCCESS == internal_mhd_readline(buffer, '=', &key_end, &value_begin)) {
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
              const size_t len = strlen(header_filename);
              if (len < filename_max_length) {
                strncpy(filename, header_filename, len);
                *header_size = ftell(file);
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
          size_t n = 0;
          while (EXIT_SUCCESS == internal_mhd_readline(value, ' ', &key_end, &value_begin)) {
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


LIBXSMM_API_INLINE int internal_mhd_read(FILE* file, void* data,
  const size_t* data_size, const size_t* size, size_t typesize,
  size_t ndims)
{
  int result = EXIT_SUCCESS;

  assert(0 != data_size);
  if (1 < ndims) {
    if (0 == size || size[0] <= data_size[0]) {
      const size_t d = ndims - 1;

      if (EXIT_SUCCESS == result && (0 == size || size[d] <= data_size[d])) {
        size_t sub_size = typesize * data_size[0], i;

        for (i = 1; i < d; ++i) {
          if (0 == size || size[i] <= data_size[i]) {
            sub_size *= data_size[i];
          }
          else {
            result = EXIT_FAILURE;
            break;
          }
        }
        for (i = 0; i < data_size[d]; ++i) {
          result = internal_mhd_read(file, data, data_size, size, typesize, d);
          if (EXIT_SUCCESS == result) {
            data = (char*)data + sub_size;
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
    const size_t n = *(0 != size ? size : data_size);
    if (*data_size < n || n != fread(data, typesize, n, file)) {
      result = EXIT_FAILURE;
    }
  }

  return result;
}


LIBXSMM_API int libxsmm_mhd_read( const char* filename,
  const size_t* data_size, const size_t* size, size_t ndims, size_t ncomponents, size_t header_size,
  libxsmm_mhd_elemtype type_stored, const libxsmm_mhd_elemtype* type_data,
  void* data, void handle_entry(void*, size_t, const void*),
  char* extension, size_t extension_size)
{
  int result = EXIT_SUCCESS;
  FILE *const file = (0 != filename && 0 != *filename &&
    0 != data_size && 0 != ndims && 0 != ncomponents &&
    LIBXSMM_MHD_ELEMTYPE_UNKNOWN != type_stored &&
    (0 == type_data || LIBXSMM_MHD_ELEMTYPE_UNKNOWN != *type_data) &&
    0 != data)
    ? fopen(filename, "rb")
    : NULL;

  if (0 != file) {
    size_t typesize;
    if (0 != header_size) {
      result = fseek(file, (long)header_size, SEEK_SET);
    }
    if (EXIT_SUCCESS == result && 0 != libxsmm_mhd_typename(type_stored, &typesize)) {
      result = internal_mhd_read(file, data, data_size, size, ncomponents * typesize, ndims);
    }
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
    if (0 == size || size[0] <= data_size[0]) {
      const size_t d = ndims - 1;

      if (EXIT_SUCCESS == result && (0 == size || size[d] <= data_size[d])) {
        size_t sub_size = typesize * data_size[0], i;

        for (i = 1; i < d; ++i) {
          if (0 == size || size[i] <= data_size[i]) {
            sub_size *= data_size[i];
          }
          else {
            result = EXIT_FAILURE;
            break;
          }
        }
        for (i = 0; i < data_size[d]; ++i) {
          result = internal_mhd_write(file, data, data_size, size, typesize, d);
          if (EXIT_SUCCESS == result) {
            data = (const char*)data + sub_size;
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
    const size_t n = *(0 != size ? size : data_size);
    if (*data_size < n || n != fwrite(data, typesize, n, file)) {
      result = EXIT_FAILURE;
    }
  }

  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_mhd_write(const char* filename,
  const size_t* data_size, const size_t* size, size_t ndims,
  size_t ncomponents, libxsmm_mhd_elemtype type, const void* data,
  const char* extension_header, const void* extension, size_t extension_size)
{
  size_t typesize = 0;
  const char *const elemname = libxsmm_mhd_typename(type, &typesize);
  FILE *const file = (0 != filename && 0 != *filename &&
      0 != data_size && 0 != ndims && 0 != ncomponents &&
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
        if (0 >= fprintf(file, " %u", (unsigned int)((0 != size && 0 < size[i]) ? size[i] : data_size[i]))) {
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
      result = internal_mhd_write(file, data, data_size, size, ncomponents * typesize, ndims);
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

