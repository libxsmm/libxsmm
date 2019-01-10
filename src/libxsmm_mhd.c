/******************************************************************************
** Copyright (c) 2009-2019, Intel Corporation                                **
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
#include "libxsmm_main.h" /* libxsmm_typesize */

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

#define LIBXSMM_MHD_MINMAX(TYPE, DATA, NELEMENTS, PMIN_INOUT, PMAX_INOUT) { \
  size_t libxsmm_mhd_minmax_index_; \
  LIBXSMM_ASSERT(NULL != (PMIN_INOUT) && NULL != (PMAX_INOUT)); \
  for (libxsmm_mhd_minmax_index_ = 0; libxsmm_mhd_minmax_index_ < (NELEMENTS); ++libxsmm_mhd_minmax_index_) { \
    TYPE libxsmm_mhd_minmax_value_; \
    LIBXSMM_ASSERT(NULL != (DATA)); \
    libxsmm_mhd_minmax_value_ = ((const TYPE*)DATA)[libxsmm_mhd_minmax_index_]; \
    if (libxsmm_mhd_minmax_value_ < *((const TYPE*)PMIN_INOUT)) { \
      *((TYPE*)PMIN_INOUT) = libxsmm_mhd_minmax_value_; \
    } \
    else if (libxsmm_mhd_minmax_value_ > *((const TYPE*)PMAX_INOUT)) { \
      *((TYPE*)PMAX_INOUT) = libxsmm_mhd_minmax_value_; \
    } \
  } \
}

#define LIBXSMM_MHD_TYPE_PROMOTE(DST_TYPE, SRC_TYPE) \
  (LIBXSMM_MHD_ELEMTYPE_I64 > (DST_TYPE) || (LIBXSMM_MHD_ELEMTYPE_U64 > (DST_TYPE) \
    ? /*dst is   signed*/(LIBXSMM_MHD_ELEMTYPE_U64 > (SRC_TYPE) ? ((SRC_TYPE) > (DST_TYPE)) : 0) \
    : /*dst is unsigned*/(LIBXSMM_MHD_ELEMTYPE_U64 > (SRC_TYPE) ? 0 : ((SRC_TYPE) > (DST_TYPE)))))

#define LIBXSMM_MHD_ELEMENT_CONVERSION(DST_TYPE, DST_ENUM, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT) { \
  const double h = (double)(0.5 - (DST_TYPE)0.5); \
  double s0 = 0, s1 = 0; \
  LIBXSMM_ASSERT_MSG(NULL != (PDST) && NULL != (PSRC), "Invalid input or outout"); \
  RESULT = EXIT_SUCCESS; \
  switch(SRC_ENUM) { \
    case LIBXSMM_MHD_ELEMTYPE_F64: { \
      const double s = *((const double*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = *((const double*)PSRC_MIN); s1 = *((const double*)PSRC_MAX); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */(0 <= s ? LIBXSMM_CLMP(s + h, DST_MIN, DST_MAX) : LIBXSMM_CLMP(s - h, DST_MIN, DST_MAX)) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_F32: { \
      const float s = *((const float*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = *((const float*)PSRC_MIN); s1 = *((const float*)PSRC_MAX); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */(0 <= s ? LIBXSMM_CLMP(s + h, DST_MIN, DST_MAX) : LIBXSMM_CLMP(s - h, DST_MIN, DST_MAX)) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_BF16: { \
      LIBXSMM_ASSERT_MSG(0, "Not implemented yet"); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_I64: { \
      const long long s = *((const long long*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = (double)(*((const long long*)PSRC_MIN)); s1 = (double)(*((const long long*)PSRC_MAX)); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */LIBXSMM_CLMP(s, DST_MIN, DST_MAX) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_I32: { \
      const int s = *((const int*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = *((const int*)PSRC_MIN); s1 = *((const int*)PSRC_MAX); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */LIBXSMM_CLMP(s, DST_MIN, DST_MAX) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_I16: { \
      const short s = *((const short*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = *((const short*)PSRC_MIN); s1 = *((const short*)PSRC_MAX); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */LIBXSMM_CLMP(s, DST_MIN, DST_MAX) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_I8: { \
      const signed char s = *((const signed char*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = *((const signed char*)PSRC_MIN); s1 = *((const signed char*)PSRC_MAX); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */LIBXSMM_CLMP(s, DST_MIN, DST_MAX) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_U64: { \
      const unsigned long long s = *((const unsigned long long*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = (double)(*((const unsigned long long*)PSRC_MIN)); s1 = (double)(*((const unsigned long long*)PSRC_MAX)); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */LIBXSMM_CLMP(s, DST_MIN, DST_MAX) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_U32: { \
      const unsigned int s = *((const unsigned int*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = *((const unsigned int*)PSRC_MIN); s1 = *((const unsigned int*)PSRC_MAX); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */LIBXSMM_CLMP(s, DST_MIN, DST_MAX) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_U16: { \
      const unsigned short s = *((const unsigned short*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = *((const unsigned short*)PSRC_MIN); s1 = *((const unsigned short*)PSRC_MAX); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */LIBXSMM_CLMP(s, DST_MIN, DST_MAX) \
          : /* promo */s); \
      } \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_U8: { \
      const unsigned char s = *((const unsigned char*)PSRC); \
      if (NULL != (PSRC_MIN)) { \
        LIBXSMM_ASSERT_MSG(NULL != (PSRC_MAX), "Invalid input range"); \
        s0 = *((const unsigned char*)PSRC_MIN); s1 = *((const unsigned char*)PSRC_MAX); \
        LIBXSMM_ASSERT_MSG(s0 <= s && s <= s1, "Invalid value range"); \
      } \
      if (s0 < s1) { /* scale */ \
        const double d = ((s - s0) * ((0 <= s1 ? (DST_MAX) : 0) - (0 <= s0 ? 0 : (DST_MIN))) / (s1 - s0) + (0 <= s0 ? 0 : (DST_MIN))); \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= d ? (d + h) : (d - h)); \
      } \
      else { /* clamp or promote */ \
        *((DST_TYPE*)PDST) = (DST_TYPE)(0 == LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) \
          ? /* clamp */LIBXSMM_CLMP(s, DST_MIN, DST_MAX) \
          : /* promo */s); \
      } \
    } break; \
    default: RESULT = EXIT_FAILURE; \
  } \
}


LIBXSMM_API const char* libxsmm_mhd_typename(libxsmm_mhd_elemtype type, size_t* typesize, const char** ctypename)
{
  const char *mhd_typename = NULL, *c_typename = NULL;
  size_t size = 0;
  switch (type) {
    case LIBXSMM_MHD_ELEMTYPE_F64:  { size = 8; mhd_typename = "MET_DOUBLE"; c_typename = "double";             } break;
    case LIBXSMM_MHD_ELEMTYPE_F32:  { size = 4; mhd_typename = "MET_FLOAT";  c_typename = "float";              } break;
    case LIBXSMM_MHD_ELEMTYPE_BF16: { size = 2; mhd_typename = "MET_BFLOAT"; c_typename = "unsigned short";     } break;
    case LIBXSMM_MHD_ELEMTYPE_I64:  { size = 8; mhd_typename = "MET_LONG";   c_typename = "signed long long";   } break;
    case LIBXSMM_MHD_ELEMTYPE_I32:  { size = 4; mhd_typename = "MET_INT";    c_typename = "signed int";         } break;
    case LIBXSMM_MHD_ELEMTYPE_I16:  { size = 2; mhd_typename = "MET_SHORT";  c_typename = "signed short";       } break;
    case LIBXSMM_MHD_ELEMTYPE_I8:   { size = 1; mhd_typename = "MET_CHAR";   c_typename = "signed char";        } break;
    case LIBXSMM_MHD_ELEMTYPE_U64:  { size = 8; mhd_typename = "MET_ULONG";  c_typename = "unsigned long long"; } break;
    case LIBXSMM_MHD_ELEMTYPE_U32:  { size = 4; mhd_typename = "MET_UINT";   c_typename = "unsigned int";       } break;
    case LIBXSMM_MHD_ELEMTYPE_U16:  { size = 2; mhd_typename = "MET_USHORT"; c_typename = "unsigned short";     } break;
    case LIBXSMM_MHD_ELEMTYPE_U8:   { size = 1; mhd_typename = "MET_UCHAR";  c_typename = "unsigned char";      } break;
    default: size = libxsmm_typesize((libxsmm_datatype)type); /* fall-back */
  }
  assert(size <= LIBXSMM_MHD_MAX_ELEMSIZE);
  if (NULL != ctypename) *ctypename = c_typename;
  if (NULL != typesize) *typesize = size;
  return mhd_typename;
}


LIBXSMM_API libxsmm_mhd_elemtype libxsmm_mhd_typeinfo(const char elemname[])
{
  libxsmm_mhd_elemtype result = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
  if (0 == strcmp("MET_DOUBLE", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_F64;
  }
  else if (0 == strcmp("MET_FLOAT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_F32;
  }
  else if (0 == strcmp("MET_BFLOAT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_BF16;
  }
  else if (0 == strcmp("MET_LONG", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_I64;
  }
  else if (0 == strcmp("MET_INT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_I32;
  }
  else if (0 == strcmp("MET_SHORT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_I16;
  }
  else if (0 == strcmp("MET_CHAR", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_I8;
  }
  else if (0 == strcmp("MET_ULONG", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_U64;
  }
  else if (0 == strcmp("MET_UINT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_U32;
  }
  else if (0 == strcmp("MET_USHORT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_U16;
  }
  else if (0 == strcmp("MET_UCHAR", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_U8;
  }
  return result;
}

LIBXSMM_API_INLINE int internal_mhd_readline(char buffer[], char split, size_t* key_end, size_t* value_begin)
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


LIBXSMM_API int libxsmm_mhd_read_header(const char header_filename[], size_t filename_max_length,
  char filename[], size_t* ndims, size_t size[], size_t* ncomponents, libxsmm_mhd_elemtype* type,
  size_t* header_size, size_t* extension_size)
{
  int result = EXIT_SUCCESS;
  char buffer[LIBXSMM_MHD_MAX_LINELENGTH];
  FILE *const file = (0 < filename_max_length && 0 != filename && 0 != ndims && 0 < *ndims && 0 != size && 0 != type && 0 != ncomponents)
                      ? fopen(header_filename, "rb") : 0;

  if (0 != file) {
    size_t key_end, value_begin;
    if (0 != extension_size) *extension_size = 0;
    if (0 != header_size) *header_size = 0;
    memset(size, 0, *ndims * sizeof(*size));
    *type = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
    *ncomponents = 1;
    if (header_filename != filename) {
      *filename = 0;
    }

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
              memcpy(filename, header_filename, len + 1);
              assert(0 == filename[len]);
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
            memcpy(filename, value, len + 1);
            assert(0 == filename[len]);
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
        while (EXIT_SUCCESS == internal_mhd_readline(value, ' ', &key_end, &value_begin) && n < *ndims) {
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
        if (EXIT_SUCCESS == result) {
          if (0 != *value && n < *ndims) {
            const int ivalue = atoi(value);
            if (0 < ivalue) {
              *isize = ivalue;
            }
            else {
              result = EXIT_FAILURE;
            }
            ++n;
          }
#if 0
          else {
            result = EXIT_FAILURE;
          }
#endif
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
    /* check size, and eventually trim dimensionality */
    if (EXIT_SUCCESS == result) {
      size_t i, d = 1;
      for (i = *ndims; 0 < i; --i) {
        if (0 != d && 1 == size[i-1]) {
          --*ndims;
        }
        else if (0 == size[i-1]) {
          result = EXIT_FAILURE;
          break;
        }
        else {
          d = 0;
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
    /* release file handle */
    if (0 != fclose(file) && EXIT_SUCCESS == result) result = EXIT_FAILURE;
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API int libxsmm_mhd_element_conversion(
  void* dst, libxsmm_mhd_elemtype dst_type, libxsmm_mhd_elemtype src_type,
  const void* src, const void* src_min, const void* src_max)
{
  int result;
  switch (dst_type) {
    case LIBXSMM_MHD_ELEMTYPE_F64: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(double, dst_type, -1.0, 1.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_F32: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(float, dst_type, -1.0, 1.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_BF16: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(libxsmm_bfloat16, dst_type, -1.0, 1.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I64: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(long long, dst_type, -9223372036854775808.0, 9223372036854775807.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I32: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(int, dst_type, -2147483648.0, 2147483647.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I16: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(short, dst_type, -32768.0, 32767.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I8: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(signed char, dst_type, -128.0, 127.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U64: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(unsigned long long, dst_type, 0.0, 18446744073709551615.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U32: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(unsigned int, dst_type, 0.0, 4294967295.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U16: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(unsigned short, dst_type, 0.0, 65535.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U8: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(unsigned char, dst_type, 0.0, 255.0, dst, src_type, src, src_min, src_max, result);
    } break;
    default: result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_mhd_element_comparison(
  void* dst, libxsmm_mhd_elemtype dst_type, libxsmm_mhd_elemtype src_type,
  const void* src, const void* src_min, const void* src_max)
{
  size_t typesize;
  int result;

  if (0 != libxsmm_mhd_typename(src_type, &typesize, NULL/*ctypename*/)) {
    if (dst_type == src_type) { /* direct comparison */
      result = libxsmm_diff(src, dst, (unsigned char)typesize);
    }
    else { /* conversion into source type */
      char element[LIBXSMM_MHD_MAX_ELEMSIZE];
      result = libxsmm_mhd_element_conversion(element, dst_type, src_type, src, src_min, src_max);
      if (EXIT_SUCCESS == result) {
        result = libxsmm_diff(src, element, (unsigned char)typesize);
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API_INLINE int internal_mhd_minmax(const void* data, size_t nelements,
  libxsmm_mhd_elemtype type, const void* minval, const void* maxval)
{
  int result;
  if ((NULL != data || 0 < nelements) && NULL != minval && NULL != maxval) {
    result = EXIT_SUCCESS;
    switch (type) {
      case LIBXSMM_MHD_ELEMTYPE_F64: {
        LIBXSMM_MHD_MINMAX(double, data, nelements, minval, maxval);              } break;
      case LIBXSMM_MHD_ELEMTYPE_F32: {
        LIBXSMM_MHD_MINMAX(float, data, nelements, minval, maxval);               } break;
      case LIBXSMM_MHD_ELEMTYPE_BF16: {
        LIBXSMM_MHD_MINMAX(libxsmm_bfloat16, data, nelements, minval, maxval);    } break;
      case LIBXSMM_MHD_ELEMTYPE_I64: {
        LIBXSMM_MHD_MINMAX(long long, data, nelements, minval, maxval);           } break;
      case LIBXSMM_MHD_ELEMTYPE_I32: {
        LIBXSMM_MHD_MINMAX(int, data, nelements, minval, maxval);                 } break;
      case LIBXSMM_MHD_ELEMTYPE_I16: {
        LIBXSMM_MHD_MINMAX(short, data, nelements, minval, maxval);               } break;
      case LIBXSMM_MHD_ELEMTYPE_I8: {
        LIBXSMM_MHD_MINMAX(signed char, data, nelements, minval, maxval);         } break;
      case LIBXSMM_MHD_ELEMTYPE_U64: {
        LIBXSMM_MHD_MINMAX(unsigned long long, data, nelements, minval, maxval);  } break;
      case LIBXSMM_MHD_ELEMTYPE_U32: {
        LIBXSMM_MHD_MINMAX(unsigned int, data, nelements, minval, maxval);        } break;
      case LIBXSMM_MHD_ELEMTYPE_U16: {
        LIBXSMM_MHD_MINMAX(unsigned short, data, nelements, minval, maxval);      } break;
      case LIBXSMM_MHD_ELEMTYPE_U8: {
        LIBXSMM_MHD_MINMAX(unsigned char, data, nelements, minval, maxval);       } break;
      default: result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_INLINE int internal_mhd_read(FILE* file, void* data, const size_t size[], const size_t pitch[],
  size_t ndims, size_t ncomponents, libxsmm_mhd_elemtype type_stored, libxsmm_mhd_elemtype type_data,
  size_t typesize, libxsmm_mhd_element_handler handle_element, int minmax, void* minval, void* maxval)
{
  int result = EXIT_SUCCESS;
  size_t typesize_stored;

  assert(0 != pitch && 0 != typesize);
  if (0 != libxsmm_mhd_typename(type_stored, &typesize_stored, NULL/*ctypename*/)) {
    if (1 < ndims) {
      if (size[0] <= pitch[0]) {
        const size_t d = ndims - 1;

        if (EXIT_SUCCESS == result) {
          if (size[d] <= pitch[d]) {
            size_t sub_size = ncomponents * typesize * pitch[0], i;

            for (i = 1; i < d; ++i) {
              if (size[i] <= pitch[i]) {
                sub_size *= pitch[i];
              }
              else {
                result = EXIT_FAILURE;
                break;
              }
            }
            for (i = 0; i < size[d] && EXIT_SUCCESS == result; ++i) {
              result = internal_mhd_read(file, data, size, pitch, d, ncomponents,
                type_stored, type_data, typesize, handle_element, minmax, minval, maxval);
              data = ((char*)data) + sub_size;
            }
          }
          else {
            result = EXIT_FAILURE;
          }
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    else if (1 == ndims) {
      if (size[0] <= pitch[0]) {
        if (type_stored == type_data && 0 == handle_element) {
          if (size[0] != fread(data, ncomponents * typesize_stored, size[0], file)) {
            result = EXIT_FAILURE;
          }
        }
        else { /* data-conversion or custom data-handler */
          const libxsmm_mhd_element_handler handler = (0 == minmax ? (0 != handle_element ? handle_element : libxsmm_mhd_element_conversion) : NULL);
          char element[LIBXSMM_MHD_MAX_ELEMSIZE];
          size_t i, j;

          for (i = 0; i < size[0]; ++i) {
            for (j = 0; j < ncomponents; ++j) {
              if (EXIT_SUCCESS == result) {
                if (1 == fread(element, typesize_stored, 1, file)) {
                  if (NULL == handler) { /* determine value-range for scaled data-conversion */
                    LIBXSMM_ASSERT(0 != minmax);
                    result = internal_mhd_minmax(element, 1/*n*/, type_stored, minval, maxval);
                  }
                  else { /* re-read data incl. conversion */
                    LIBXSMM_ASSERT(0 == minmax);
                    result = handler(data, type_data, type_stored, element, minval, maxval);
                    data = ((char*)data) + typesize;
                  }
                }
                else {
                  result = EXIT_FAILURE;
                }
              }
              else {
                i = size[0]; /* break outer */
                break;
              }
            }
          }
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API int libxsmm_mhd_read(const char filename[],
  const size_t offset[], const size_t size[], const size_t pitch[], size_t ndims, size_t ncomponents,
  size_t header_size, libxsmm_mhd_elemtype type_stored, const libxsmm_mhd_elemtype* type_data,
  void* data, libxsmm_mhd_element_handler handle_element, char extension[], size_t extension_size)
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
    const libxsmm_mhd_elemtype datatype = (type_data ? *type_data : type_stored);
    const size_t *const shape = (0 != pitch ? pitch : size);
    size_t offset1 = (0 != offset ? offset[0] : 0), typesize = 0, i;

    /* check that size is less-equal than pitch */
    if (EXIT_SUCCESS == result) {
      for (i = 0; i < ndims; ++i) {
        if (size[i] > shape[i]) {
          result = EXIT_FAILURE;
          break;
        }
      }
    }
    /* zeroing buffer if pitch is larger than size */
    if (EXIT_SUCCESS == result) {
      if (0 != libxsmm_mhd_typename(datatype, &typesize, NULL/*ctypename*/)) {
        size_t size1 = size[0], pitch1 = shape[0];
        for (i = 1; i < ndims; ++i) {
          offset1 += (0 != offset ? offset[i] : 0) * pitch1;
          pitch1 *= shape[i];
          size1 *= size[i];
        }
        assert(size1 <= pitch1);
        if (size1 != pitch1 && 0 == handle_element) {
          memset(data, 0, pitch1 * ncomponents * typesize);
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      char minmax[2*(LIBXSMM_MHD_MAX_ELEMSIZE)];
      if (0 != header_size) result = fseek(file, (long)header_size, SEEK_SET); /* set file position to data section */
      if (EXIT_SUCCESS == result) {
        if (1 == fread(minmax, typesize, 1, file)) {
          memcpy(minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), minmax, typesize);
          result = fseek(file, (long)header_size, SEEK_SET); /* reset file position */
        }
        else {
          result = EXIT_FAILURE;
        }
      }
      if (EXIT_SUCCESS == result) {
        result = internal_mhd_read(file, ((char*)data) + offset1 * ncomponents * typesize, size, shape,
          ndims, ncomponents, type_stored, datatype, typesize, handle_element,
          1/*search min-max*/, minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE));
      }
      if (EXIT_SUCCESS == result && 0 != libxsmm_diff(minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), LIBXSMM_MHD_MAX_ELEMSIZE)) {
        if (EXIT_SUCCESS == result) result = fseek(file, (long)header_size, SEEK_SET); /* reset file position */
        if (EXIT_SUCCESS == result) {
          result = internal_mhd_read(file, ((char*)data) + offset1 * ncomponents * typesize, size, shape,
            ndims, ncomponents, type_stored, datatype, typesize, handle_element,
            0/*use min-max*/, minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE));
        }
      }
    }
    if (0 != extension && 0 < extension_size) {
      if (extension_size != fread(extension, 1, extension_size, file)) {
        result = EXIT_FAILURE;
      }
    }
    /* release file handle */
    if (0 != fclose(file) && EXIT_SUCCESS == result) result = EXIT_FAILURE;
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API_INLINE int internal_mhd_write(FILE* file, const void* data, const size_t size[], const size_t pitch[],
  size_t ndims, size_t ncomponents, libxsmm_mhd_elemtype type_data, libxsmm_mhd_elemtype type,
  size_t typesize_data, size_t typesize, int minmax, void* minval, void* maxval)
{
  int result = EXIT_SUCCESS;

  assert(0 != pitch);
  if (1 < ndims) {
    if (size[0] <= pitch[0]) {
      const size_t d = ndims - 1;

      if (EXIT_SUCCESS == result) {
        if (size[d] <= pitch[d]) {
          size_t sub_size = ncomponents * typesize_data * pitch[0], i;

          for (i = 1; i < d; ++i) {
            if (size[i] <= pitch[i]) {
              sub_size *= pitch[i];
            }
            else {
              result = EXIT_FAILURE;
              break;
            }
          }
          for (i = 0; i < size[d] && EXIT_SUCCESS == result; ++i) {
            result = internal_mhd_write(file, data, size, pitch, d, ncomponents,
              type_data, type, typesize_data, typesize, minmax, minval, maxval);
            data = ((const char*)data) + sub_size;
          }
        }
        else {
          result = EXIT_FAILURE;
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  else if (1 == ndims) {
    if (size[0] <= pitch[0]) {
      if (type == type_data) {
        if (size[0] != fwrite(data, ncomponents * typesize_data, size[0], file)) {
          result = EXIT_FAILURE;
        }
      }
      else { /* data-conversion */
        char element[LIBXSMM_MHD_MAX_ELEMSIZE];
        size_t i, j;

        if (0 != minmax) {
          /* determine value-range for scaled data-conversion */
          memcpy(minval, data, typesize_data); memcpy(maxval, data, typesize_data); /* initial condition */
          result = internal_mhd_minmax(data, size[0] * ncomponents, type_data, minval, maxval);
        }
        else {
          for (i = 0; i < size[0]; ++i) {
            for (j = 0; j < ncomponents; ++j) {
              if (EXIT_SUCCESS == result) {
                result = libxsmm_mhd_element_conversion(element, type, type_data, data, minval, maxval);
                if (EXIT_SUCCESS == result) {
                  if (1 == fwrite(element, typesize, 1, file)) {
                    data = ((char*)data) + typesize_data;
                  }
                  else {
                    result = EXIT_FAILURE;
                  }
                }
              }
              else {
                i = size[0]; /* break outer */
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
  }

  return result;
}


LIBXSMM_API int libxsmm_mhd_write(const char filename[],
  const size_t offset[], const size_t size[], const size_t pitch[], size_t ndims, size_t ncomponents,
  libxsmm_mhd_elemtype type_data, const libxsmm_mhd_elemtype* type, const void* data, size_t* header_size,
  const char extension_header[], const void* extension, size_t extension_size)
{
  size_t typesize = 0;
  const libxsmm_mhd_elemtype elemtype = (NULL == type ? type_data : *type);
  const char *const elemname = libxsmm_mhd_typename(elemtype, &typesize, NULL/*ctypename*/);
  FILE *const file = (0 != filename && 0 != *filename &&
    NULL != size && 0 != ndims && 0 != ncomponents && NULL != data && NULL != elemname && 0 < typesize)
    ? fopen(filename, "wb")
    : NULL;
  int result = EXIT_SUCCESS;

  if (0 != file) {
    size_t typesize_data = 0, i;
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
    if (EXIT_SUCCESS == result) {
      if (0 < fprintf(file, "\nElementSpacing =")) {
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
    /* source data type is not required to have MHD element name (type-size is needed) */
    if (EXIT_SUCCESS == result) {
      libxsmm_mhd_typename(type_data, &typesize_data, NULL/*ctypename*/);
      if (0 == typesize_data) result = EXIT_FAILURE;
    }
    /* ElementDataFile must be the last entry before writing the data */
    if (EXIT_SUCCESS == result && 0 < fprintf(file, "\nElementType = %s\nElementDataFile = LOCAL\n", elemname)) {
      const long file_position = ftell(file); /* determine the header size */
      if (0 < file_position) {
        const size_t *const shape = (0 != pitch ? pitch : size);
        char minmax[2*(LIBXSMM_MHD_MAX_ELEMSIZE)];
        result = internal_mhd_write(file,
          ((const char*)data) + libxsmm_offset(offset, shape, ndims, NULL/*size*/) * ncomponents * typesize_data,
          size, shape, ndims, ncomponents, type_data, elemtype, typesize_data, typesize,
          1/*search min-max*/, minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE));
        if (EXIT_SUCCESS == result) {
          if (0 != header_size) *header_size = file_position;
          if (0 != libxsmm_diff(minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), LIBXSMM_MHD_MAX_ELEMSIZE)) {
            result = fseek(file, file_position, SEEK_SET); /* reset file position */
            if (EXIT_SUCCESS == result) {
              result = internal_mhd_write(file,
                ((const char*)data) + libxsmm_offset(offset, shape, ndims, NULL/*size*/) * ncomponents * typesize_data,
                size, shape, ndims, ncomponents, type_data, elemtype, typesize_data, typesize,
                0/*use min-max*/, minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE));
            }
          }
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    /* append the extension data after the regular data section */
    if (EXIT_SUCCESS == result && 0 < extension_size) {
      if (extension_size != fwrite(extension, 1, extension_size, file)) {
        result = EXIT_FAILURE;
      }
    }
    /* release file handle */
    if (0 != fclose(file) && EXIT_SUCCESS == result) result = EXIT_FAILURE;
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}

