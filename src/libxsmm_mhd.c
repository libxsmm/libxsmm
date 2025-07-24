/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <utils/libxsmm_mhd.h>
#include <libxsmm_memory.h>
#include <ctype.h>

#if !defined(LIBXSMM_MHD_MAX_LINELENGTH)
# define LIBXSMM_MHD_MAX_LINELENGTH 1024
#endif

#if !defined(LIBXSMM_MHD_MAX_ELEMSIZE)
# define LIBXSMM_MHD_MAX_ELEMSIZE 8
#endif

#define LIBXSMM_MHD_MINMAX(TYPE, DATA, NELEMENTS, PMIN_INOUT, PMAX_INOUT) do { \
  LIBXSMM_ASSERT(NULL != (PMIN_INOUT) && NULL != (PMAX_INOUT)); \
  if (0 < (NELEMENTS)) { \
    size_t libxsmm_mhd_minmax_index_ = 0; \
    do { \
      TYPE libxsmm_mhd_minmax_value_; \
      LIBXSMM_ASSERT(NULL != (DATA)); \
      libxsmm_mhd_minmax_value_ = ((const TYPE*)DATA)[libxsmm_mhd_minmax_index_]; \
      if (libxsmm_mhd_minmax_value_ < *((const TYPE*)PMIN_INOUT)) { \
        *((TYPE*)PMIN_INOUT) = libxsmm_mhd_minmax_value_; \
      } \
      else if (libxsmm_mhd_minmax_value_ > *((const TYPE*)PMAX_INOUT)) { \
        *((TYPE*)PMAX_INOUT) = libxsmm_mhd_minmax_value_; \
      } \
      ++libxsmm_mhd_minmax_index_; \
    } while (libxsmm_mhd_minmax_index_ < (NELEMENTS)); \
  } \
  else *((TYPE*)PMIN_INOUT) = *((TYPE*)PMAX_INOUT) = 0; \
} while(0)

#define LIBXSMM_MHD_TYPE_IS_FLOAT(ENUM) (LIBXSMM_MHD_ELEMTYPE_I64 > (ENUM))
#define LIBXSMM_MHD_TYPE_IS_UINT(ENUM) ( \
    LIBXSMM_MHD_ELEMTYPE_UNKNOWN > (ENUM) && !LIBXSMM_MHD_TYPE_IS_FLOAT(ENUM) && ( \
    LIBXSMM_MHD_ELEMTYPE_U64 == (ENUM) || LIBXSMM_MHD_ELEMTYPE_U32 == (ENUM) || \
    LIBXSMM_MHD_ELEMTYPE_U16 == (ENUM) || LIBXSMM_MHD_ELEMTYPE_U8 == (ENUM)) \
  )
#define LIBXSMM_MHD_TYPE_PROMOTE(DST_ENUM, SRC_ENUM) ( \
    (LIBXSMM_MHD_TYPE_IS_FLOAT(DST_ENUM) || (SRC_ENUM) == (DST_ENUM)) || \
    (LIBXSMM_MHD_TYPE_IS_UINT(DST_ENUM) \
      ? (LIBXSMM_MHD_TYPE_IS_UINT(SRC_ENUM) ? (libxsmm_mhd_typesize(SRC_ENUM) <= libxsmm_mhd_typesize(DST_ENUM)) : 0) \
      : (LIBXSMM_MHD_TYPE_IS_UINT(SRC_ENUM) ? 0 : (libxsmm_mhd_typesize(SRC_ENUM) <= libxsmm_mhd_typesize(DST_ENUM)))) \
  )

#define LIBXSMM_MHD_ELEMENT_CONVERSION_F(SRC_TYPE, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT) do { \
  const libxsmm_mhd_elemtype dst_enum = (NULL == (DST_INFO) ? (SRC_ENUM) : (DST_INFO)->type); \
  const double h = (0.5 - (DST_TYPE)0.5); \
  SRC_TYPE s = *((const SRC_TYPE*)PSRC); \
  double s0 = 0, s1 = 0; \
  if (NULL != (PSRC_MIN) && LIBXSMM_NOTNAN(s)) { \
    assert(NULL != (PSRC_MAX)); \
    s0 = (double)*((const SRC_TYPE*)PSRC_MIN); s1 = (double)*((const SRC_TYPE*)PSRC_MAX); \
    assert(s0 <= s1); \
  } \
  if (LIBXSMM_MHD_ELEMTYPE_I64 <= dst_enum && s0 < s1) { /* scale (integer-type) */ \
    if (LIBXSMM_MHD_TYPE_IS_UINT(dst_enum)) { \
      const double s0pos = LIBXSMM_MAX(0, s0), s1pos = LIBXSMM_MAX(0, s1), scale = (s0pos < s1pos ? ((s1 - s0) / (s1pos - s0pos)) : 1); \
      s = (SRC_TYPE)(scale * (double)LIBXSMM_MAX(0, s)); \
      s0 = s0pos; s1 = s1pos; \
    } \
    else if (0 == LIBXSMM_MHD_TYPE_PROMOTE(dst_enum, SRC_ENUM) && 0 > s0 && 0 < s1) { \
      s1 = LIBXSMM_MAX(-s0, s1); s0 = -s1; \
    } \
    { const double d0 = (0 <= s0 ? 0 : (DST_MIN)), d1 = (0 <= s1 ? (DST_MAX) : 0), d = ((double)s - s0) * (d1 - d0) / (s1 - s0) + d0; \
      *((DST_TYPE*)PDST) = (DST_TYPE)LIBXSMM_CLMP(0 <= d ? (d + h) : (d - h), d0, d1); \
    } \
  } \
  else if (LIBXSMM_MHD_TYPE_IS_UINT(dst_enum) && NULL != (DST_INFO) \
    && LIBXSMM_MHD_ELEMENT_CONVERSION_MODULUS == (DST_INFO)->hint) \
  { /* hint */ \
    const double d = (DST_MAX) - (DST_MIN) + 1, q = s / d; \
    *((DST_TYPE*)PDST) = (DST_TYPE)(s - d * (DST_TYPE)q); \
  } \
  else if (0 == LIBXSMM_MHD_TYPE_PROMOTE(dst_enum, SRC_ENUM)) { /* clamp */ \
    *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= s ? LIBXSMM_CLMP(s + h, DST_MIN, DST_MAX) : LIBXSMM_CLMP(s - h, DST_MIN, DST_MAX)); \
  } \
  else { /* promote */ \
    *((DST_TYPE*)PDST) = (DST_TYPE)(0 <= s ? (s + h) : (s - h)); \
  } \
  RESULT = EXIT_SUCCESS; \
} while(0)

#define LIBXSMM_MHD_ELEMENT_CONVERSION_I(SRC_TYPE, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT) do { \
  const libxsmm_mhd_elemtype dst_enum = (NULL == (DST_INFO) ? (SRC_ENUM) : (DST_INFO)->type); \
  const double h = (0.5 - (DST_TYPE)0.5); \
  SRC_TYPE s = *((const SRC_TYPE*)PSRC); \
  double s0 = 0, s1 = 0; \
  if (NULL != (PSRC_MIN)) { \
    assert(NULL != (PSRC_MAX)); \
    s0 = (double)*((const SRC_TYPE*)PSRC_MIN); s1 = (double)*((const SRC_TYPE*)PSRC_MAX); \
    assert(s0 <= s1); \
  } \
  if (LIBXSMM_MHD_ELEMTYPE_I64 <= dst_enum && s0 < s1) { /* scale (integer-type) */ \
    if (LIBXSMM_MHD_TYPE_IS_UINT(dst_enum)) { \
      const double s0pos = LIBXSMM_MAX(0, s0), s1pos = LIBXSMM_MAX(0, s1), scale = (s0pos < s1pos ? ((s1 - s0) / (s1pos - s0pos)) : 1); \
      const double ss = scale * (double)LIBXSMM_MAX(0, s); \
      s = (SRC_TYPE)(0 <= ss ? (ss + h) : (ss - h)); \
      s0 = s0pos; s1 = s1pos; \
    } \
    else if (0 == LIBXSMM_MHD_TYPE_PROMOTE(dst_enum, SRC_ENUM) && 0 > s0 && 0 < s1) { \
      s1 = LIBXSMM_MAX(-s0, s1); s0 = -s1; \
    } \
    { const double d0 = (0 <= s0 ? 0 : (DST_MIN)), d1 = (0 <= s1 ? (DST_MAX) : 0), d = ((double)s - s0) * (d1 - d0) / (s1 - s0) + d0; \
      *((DST_TYPE*)PDST) = (DST_TYPE)LIBXSMM_CLMP(0 <= d ? (d + h) : (d - h), d0, d1); \
    } \
  } \
  else if (LIBXSMM_MHD_TYPE_IS_UINT(dst_enum) && NULL != (DST_INFO) \
    && LIBXSMM_MHD_ELEMENT_CONVERSION_MODULUS == (DST_INFO)->hint) \
  { /* hint */ \
    const double d = (DST_MAX) - (DST_MIN) + 1, q = s / d; \
    *((DST_TYPE*)PDST) = (DST_TYPE)(s - d * (DST_TYPE)q); \
  } \
  else if (0 == LIBXSMM_MHD_TYPE_PROMOTE(dst_enum, SRC_ENUM)) { /* clamp */ \
    *((DST_TYPE*)PDST) = (DST_TYPE)LIBXSMM_CLMP(s, DST_MIN, DST_MAX); \
  } \
  else { /* promote */ \
    *((DST_TYPE*)PDST) = (DST_TYPE)s; \
  } \
  RESULT = EXIT_SUCCESS; \
} while(0)

#define LIBXSMM_MHD_ELEMENT_CONVERSION_U LIBXSMM_MHD_ELEMENT_CONVERSION_I

#define LIBXSMM_MHD_ELEMENT_CONVERSION(DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT) do { \
  LIBXSMM_ASSERT_MSG(NULL != (PDST) && NULL != (PSRC), "Invalid input or output"); \
  switch((int)(SRC_ENUM)) { \
    case LIBXSMM_MHD_ELEMTYPE_I64: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_I(long long, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_I32: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_I(int, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_I16: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_I(short, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_I8: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_I(signed char, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_U64: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_U(unsigned long long, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_U32: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_U(unsigned int, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_U16: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_U(unsigned short, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_U8: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_U(unsigned char, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_F64: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_F(double, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_F32: { \
      LIBXSMM_MHD_ELEMENT_CONVERSION_F(float, DST_TYPE, DST_INFO, DST_MIN, DST_MAX, PDST, SRC_ENUM, PSRC, PSRC_MIN, PSRC_MAX, RESULT); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_F16: { \
      LIBXSMM_ASSERT_MSG(0, "Not implemented yet"); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_BF16: { \
      LIBXSMM_ASSERT_MSG(0, "Not implemented yet"); \
    } break; \
    case LIBXSMM_MHD_ELEMTYPE_BF8: { \
      LIBXSMM_ASSERT_MSG(0, "Not implemented yet"); \
    } break; \
    default: RESULT = EXIT_FAILURE; \
  } \
} while(0)


LIBXSMM_API const char* libxsmm_mhd_typename(libxsmm_mhd_elemtype type, const char** ctypename)
{
  const char *mhd_typename = NULL, *c_typename = NULL;
  switch ((int)type) {
    case LIBXSMM_MHD_ELEMTYPE_F64:  { mhd_typename = "MET_DOUBLE";  c_typename = "double";             } break;
    case LIBXSMM_MHD_ELEMTYPE_F32:  { mhd_typename = "MET_FLOAT";   c_typename = "float";              } break;
    case LIBXSMM_MHD_ELEMTYPE_F16:  { mhd_typename = "MET_HALF";    c_typename = "unsigned short";     } break;
    case LIBXSMM_MHD_ELEMTYPE_BF16: { mhd_typename = "MET_BFLOAT";  c_typename = "unsigned short";     } break;
    case LIBXSMM_MHD_ELEMTYPE_BF8:  { mhd_typename = "MET_BFLOAT8"; c_typename = "unsigned char";      } break;
    case LIBXSMM_MHD_ELEMTYPE_I64:  { mhd_typename = "MET_LONG";    c_typename = "signed long long";   } break;
    case LIBXSMM_MHD_ELEMTYPE_I32:  { mhd_typename = "MET_INT";     c_typename = "signed int";         } break;
    case LIBXSMM_MHD_ELEMTYPE_I16:  { mhd_typename = "MET_SHORT";   c_typename = "signed short";       } break;
    case LIBXSMM_MHD_ELEMTYPE_I8:   { mhd_typename = "MET_CHAR";    c_typename = "signed char";        } break;
    case LIBXSMM_MHD_ELEMTYPE_U64:  { mhd_typename = "MET_ULONG";   c_typename = "unsigned long long"; } break;
    case LIBXSMM_MHD_ELEMTYPE_U32:  { mhd_typename = "MET_UINT";    c_typename = "unsigned int";       } break;
    case LIBXSMM_MHD_ELEMTYPE_U16:  { mhd_typename = "MET_USHORT";  c_typename = "unsigned short";     } break;
    case LIBXSMM_MHD_ELEMTYPE_U8:   { mhd_typename = "MET_UCHAR";   c_typename = "unsigned char";      } break;
    default: LIBXSMM_ASSERT_MSG(0, "Unknown type"); /* fallback */
  }
  if (NULL != ctypename) *ctypename = c_typename;
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
  else if (0 == strcmp("MET_HALF", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_F16;
  }
  else if (0 == strcmp("MET_BFLOAT", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_BF16;
  }
  else if (0 == strcmp("MET_BFLOAT8", elemname)) {
    result = LIBXSMM_MHD_ELEMTYPE_BF8;
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


LIBXSMM_API size_t libxsmm_mhd_typesize(libxsmm_mhd_elemtype type)
{
  const size_t result = libxsmm_typesize((libxsmm_datatype)type);
  LIBXSMM_ASSERT(result <= LIBXSMM_MHD_MAX_ELEMSIZE);
  return result;
}


LIBXSMM_API_INLINE int internal_mhd_readline(char buffer[], char split, size_t* key_end, size_t* value_begin)
{
  int result = EXIT_SUCCESS;
  char *const isplit = strchr(buffer, split);

  if (NULL != isplit) {
    char* i = isplit;
    LIBXSMM_ASSERT(NULL != key_end && NULL != value_begin);
    while (buffer != i) { --i;  if (0 == isspace((int)(*i))) break; }
    *key_end = i - buffer + 1;
    i = isplit;
    while ('\n' != *++i) if (0 == isspace((int)(*i))) break;
    *value_begin = i - buffer;
    while (0 != *i && 0 != isprint((int)(*i))) ++i;
    if (0 == isprint((int)(*i))) *i = 0; /* fix-up */
    if (i <= (buffer + *value_begin)) result = EXIT_FAILURE;
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
  FILE *const file = (
      0 < filename_max_length && NULL != filename &&
      NULL != ndims && 0 < *ndims && NULL != size &&
      NULL != type && NULL != ncomponents)
    ? fopen(header_filename, "rb") : NULL;
  if (NULL != file) {
    size_t key_end, value_begin;
    if (NULL != extension_size) *extension_size = 0;
    if (NULL != header_size) *header_size = 0;
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
        if (0 < value) *ncomponents = value;
        else result = EXIT_FAILURE;
      }
      else if (NULL != extension_size
        && 0 == strncmp("ExtensionDataSize", buffer, key_end)
        && key_end == strlen("ExtensionDataSize"))
      {
        const int value = atoi(buffer + value_begin);
        if (0 <= value) *extension_size = value;
        else result = EXIT_FAILURE;
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
            const long file_position = ftell(file); /* determine the header size */
            const size_t len = strlen(header_filename);
            if (0 <= file_position && len < filename_max_length) {
              memcpy(filename, header_filename, len + 1);
              LIBXSMM_ASSERT(0 == filename[len]);
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
            LIBXSMM_ASSERT(0 == filename[len]);
          }
          else result = EXIT_FAILURE;
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
        }
      }
      else if (0 == strncmp("BinaryData", buffer, key_end)
        && key_end == strlen("BinaryData"))
      {
        const char *const value = buffer + value_begin;
        if (0 == strcmp("False", value) || 0 != strcmp("True", value)) result = EXIT_FAILURE;
      }
      else if (0 == strncmp("CompressedData", buffer, key_end)
        && key_end == strlen("CompressedData"))
      {
        const char *const value = buffer + value_begin;
        if (0 == strcmp("True", value) || 0 != strcmp("False", value)) result = EXIT_FAILURE;
      }
      else if ((0 == strncmp("BinaryDataByteOrderMSB", buffer, key_end) && key_end == strlen("BinaryDataByteOrderMSB"))
            || (0 == strncmp("ElementByteOrderMSB",    buffer, key_end) && key_end == strlen("ElementByteOrderMSB")))
      {
        const char *const value = buffer + value_begin;
        if (0 == strcmp("True", value) || 0 != strcmp("False", value)) result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result && (0 == *filename || LIBXSMM_MHD_ELEMTYPE_UNKNOWN == *type)) result = EXIT_FAILURE;
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
        else d = 0;
      }
    }
    /* prefix the path of the header file to make sure that the data file can be found */
    if (EXIT_SUCCESS == result && (NULL == header_size || 0 == *header_size)) {
      const char* split = header_filename + strlen(header_filename) - 1;
      while (header_filename != split && NULL == strchr("/\\", *split)) --split;
      if (header_filename < split) {
        const size_t len = strlen(filename), n = split - header_filename + 1;
        if ((len+ n) <= filename_max_length) {
          size_t i;
          for (i = 1; i <= len; ++i) filename[len + n - i] = filename[len - i];
          for (i = 0; i < n; ++i) filename[i] = header_filename[i];
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


LIBXSMM_API int libxsmm_mhd_element_conversion(void* dst, const libxsmm_mhd_element_handler_info* dst_info,
  libxsmm_mhd_elemtype src_type, const void* src, const void* src_min, const void* src_max)
{
  const libxsmm_mhd_elemtype dst_type = (NULL == dst_info ? src_type : dst_info->type);
  int result = EXIT_SUCCESS;
  switch ((int)dst_type) {
    case LIBXSMM_MHD_ELEMTYPE_F64: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(double, dst_info, -1.0, 1.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_F32: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(float, dst_info, -1.0, 1.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_F16: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(libxsmm_float16, dst_info, -1.0, 1.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_BF16: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(libxsmm_bfloat16, dst_info, -1.0, 1.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_BF8: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(libxsmm_bfloat8, dst_info, -1.0, 1.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I64: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(long long, dst_info, -9223372036854775808.0, 9223372036854775807.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I32: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(int, dst_info, -2147483648.0, 2147483647.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I16: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(short, dst_info, -32768.0, 32767.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_I8: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(signed char, dst_info, -128.0, 127.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U64: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(unsigned long long, dst_info, 0.0, 18446744073709551615.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U32: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(unsigned int, dst_info, 0.0, 4294967295.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U16: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(unsigned short, dst_info, 0.0, 65535.0, dst, src_type, src, src_min, src_max, result);
    } break;
    case LIBXSMM_MHD_ELEMTYPE_U8: {
      LIBXSMM_MHD_ELEMENT_CONVERSION(unsigned char, dst_info, 0.0, 255.0, dst, src_type, src, src_min, src_max, result);
    } break;
    default: result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_mhd_element_comparison(void* dst, const libxsmm_mhd_element_handler_info* dst_info,
  libxsmm_mhd_elemtype src_type, const void* src, const void* src_min, const void* src_max)
{
  const size_t typesize = libxsmm_mhd_typesize(src_type);
  int result;
  if (NULL == dst_info || dst_info->type == src_type) { /* direct comparison */
    result = libxsmm_diff(src, dst, (unsigned char)typesize);
  }
  else { /* conversion into source type */
    char element[LIBXSMM_MHD_MAX_ELEMSIZE];
    result = libxsmm_mhd_element_conversion(element, dst_info,
      src_type, src, src_min, src_max);
    if (EXIT_SUCCESS == result) {
      result = libxsmm_diff(src, element, (unsigned char)typesize);
    }
  }
  return result;
}


/* coverity[var_deref_op] */
LIBXSMM_API_INLINE int internal_mhd_minmax(const void* data, size_t nelements,
  libxsmm_mhd_elemtype type, void* minval, void* maxval)
{
  int result;
  if ((NULL != data || 0 == nelements) && NULL != minval && NULL != maxval) {
    result = EXIT_SUCCESS;
    switch ((int)type) {
      case LIBXSMM_MHD_ELEMTYPE_F64: {
        LIBXSMM_MHD_MINMAX(double, data, nelements, minval, maxval);              } break;
      case LIBXSMM_MHD_ELEMTYPE_F32: {
        LIBXSMM_MHD_MINMAX(float, data, nelements, minval, maxval);               } break;
      case LIBXSMM_MHD_ELEMTYPE_F16: {
        LIBXSMM_MHD_MINMAX(libxsmm_float16, data, nelements, minval, maxval);     } break;
      case LIBXSMM_MHD_ELEMTYPE_BF16: {
        LIBXSMM_MHD_MINMAX(libxsmm_bfloat16, data, nelements, minval, maxval);    } break;
      case LIBXSMM_MHD_ELEMTYPE_BF8: {
        LIBXSMM_MHD_MINMAX(libxsmm_bfloat8, data, nelements, minval, maxval);     } break;
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


LIBXSMM_API_INTERN int internal_mhd_read(FILE* /*file*/, void* /*data*/, const size_t /*size*/[], const size_t /*pitch*/[],
  size_t /*ndims*/, size_t /*ncomponents*/, libxsmm_mhd_elemtype /*type_stored*/, size_t /*typesize*/,
  const libxsmm_mhd_element_handler_info* /*handler_info*/, libxsmm_mhd_element_handler /*handler*/,
  void* /*minval*/, void* /*maxval*/, int /*minmax*/);
LIBXSMM_API_INTERN int internal_mhd_read(FILE* file, void* data, const size_t size[], const size_t pitch[],
  size_t ndims, size_t ncomponents, libxsmm_mhd_elemtype type_stored, size_t typesize,
  const libxsmm_mhd_element_handler_info* handler_info, libxsmm_mhd_element_handler handler,
  void* minval, void* maxval, int minmax)
{
  const size_t typesize_stored = libxsmm_mhd_typesize(type_stored);
  int result = EXIT_SUCCESS;
  LIBXSMM_ASSERT(NULL != pitch && 0 != typesize);
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
              type_stored, typesize, handler_info, handler, minval, maxval, minmax);
            data = ((char*)data) + sub_size;
          }
        }
        else result = EXIT_FAILURE;
      }
    }
    else result = EXIT_FAILURE;
  }
  else if (1 == ndims) {
    if (size[0] <= pitch[0]) {
      if ((NULL == handler_info || type_stored == handler_info->type) && NULL == handler) { /* fast-path */
        if (size[0] != fread(data, ncomponents * typesize_stored, size[0], file)) result = EXIT_FAILURE;
      }
      else { /* data-conversion or custom data-handler */
        const libxsmm_mhd_element_handler handle_element = (0 == minmax
          ? (NULL != handler ? handler : libxsmm_mhd_element_conversion)
          : (NULL));
        char element[LIBXSMM_MHD_MAX_ELEMSIZE];
        size_t i, j;
        for (i = 0; i < size[0]; ++i) {
          for (j = 0; j < ncomponents; ++j) {
            if (EXIT_SUCCESS == result) {
              if (1 == fread(element, typesize_stored, 1, file)) {
                if (NULL == handle_element) { /* determine value-range for scaled data-conversion */
                  LIBXSMM_ASSERT(0 != minmax);
                  result = internal_mhd_minmax(element, 1/*n*/, type_stored, minval, maxval);
                }
                else { /* re-read data incl. conversion */
                  LIBXSMM_ASSERT(0 == minmax);
                  result = handle_element(data, handler_info, type_stored, element, minval, maxval);
                  data = ((char*)data) + typesize;
                }
              }
              else result = EXIT_FAILURE;
            }
            else {
              i = size[0]; /* break outer */
              break;
            }
          }
        }
      }
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_mhd_read(const char filename[],
  const size_t offset[], const size_t size[], const size_t pitch[], size_t ndims,
  size_t ncomponents, size_t header_size, libxsmm_mhd_elemtype type_stored, void* data,
  const libxsmm_mhd_element_handler_info* handler_info, libxsmm_mhd_element_handler handler,
  char extension[], size_t extension_size)
{
  int result = EXIT_SUCCESS;
  const libxsmm_mhd_elemtype datatype = (NULL == handler_info ? type_stored : handler_info->type);
  FILE *const file = ((NULL != filename && 0 != *filename && NULL != data &&
      NULL != size && 0 != ndims && 0 != ncomponents &&
      LIBXSMM_MHD_ELEMTYPE_UNKNOWN != type_stored &&
      LIBXSMM_MHD_ELEMTYPE_UNKNOWN != datatype)
    ? fopen(filename, "rb")
    : NULL);
  if (NULL != file) {
    size_t pitch1 = 0, size1 = 0;
    const size_t *const shape = (NULL != pitch ? pitch : size);
    const size_t offset1 = libxsmm_offset(offset, shape, ndims, &pitch1);
    const size_t typesize = libxsmm_mhd_typesize(datatype);
    size_t i;
    LIBXSMM_EXPECT(0 == libxsmm_offset(NULL, size, ndims, &size1));
    result = ((offset1 + size1) <= pitch1 ? EXIT_SUCCESS : EXIT_FAILURE);
    /* check that size is less-equal than pitch */
    if (EXIT_SUCCESS == result) {
      for (i = 0; i < ndims; ++i) if (size[i] > shape[i]) {
        result = EXIT_FAILURE;
        break;
      }
    }
    /* zeroing buffer if pitch is larger than size */
    if (EXIT_SUCCESS == result && size1 != pitch1 && NULL == handler) {
      memset(data, 0, pitch1 * ncomponents * typesize);
    }
    if (EXIT_SUCCESS == result) {
      char *const output = ((char*)data) + offset1 * ncomponents * typesize;
      char minmax[2*(LIBXSMM_MHD_MAX_ELEMSIZE)];
      if (0 != header_size) result = fseek(file, (long)header_size, SEEK_SET); /* set position to data section */
      if (EXIT_SUCCESS == result && (NULL != handler /* slow-path */
        || (datatype != type_stored && LIBXSMM_MHD_ELEMENT_CONVERSION_DEFAULT == handler_info->hint)))
      { /* conversion needed */
        if (1 == fread(minmax, typesize, 1, file)) {
          LIBXSMM_ASSERT(typesize <= (LIBXSMM_MHD_MAX_ELEMSIZE));
          LIBXSMM_MEMCPY127(minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), minmax, typesize);
          result = fseek(file, (long)header_size, SEEK_SET); /* reset file position */
          if (EXIT_SUCCESS == result) {
            result = internal_mhd_read(file, NULL/*output*/, size, shape,
              ndims, ncomponents, type_stored, typesize, handler_info, handler,
              minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), 1/*search min-max*/);
          }
          if (EXIT_SUCCESS == result) {
            result = fseek(file, (long)header_size, SEEK_SET); /* reset file position */
          }
        }
        else result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        result = internal_mhd_read(file, output, size, shape,
          ndims, ncomponents, type_stored, typesize, handler_info, handler,
          minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), 0/*use min-max*/);
      }
    }
    if (NULL != extension && 0 < extension_size && extension_size != fread(extension, 1, extension_size, file)) {
      result = EXIT_FAILURE;
    }
    /* release file handle */
    if (0 != fclose(file) && EXIT_SUCCESS == result) result = EXIT_FAILURE;
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_INTERN int internal_mhd_write(FILE* /*file*/, const void* /*data*/, const size_t /*size*/[], const size_t /*pitch*/[],
  size_t /*ndims*/, size_t /*ncomponents*/, libxsmm_mhd_elemtype /*type_data*/, size_t /*typesize_data*/,
  const libxsmm_mhd_element_handler_info* /*handler_info*/, libxsmm_mhd_element_handler /*handler*/,
  void* /*minval*/, void* /*maxval*/, int /*minmax*/);
LIBXSMM_API_INTERN int internal_mhd_write(FILE* file, const void* data, const size_t size[], const size_t pitch[],
  size_t ndims, size_t ncomponents, libxsmm_mhd_elemtype type_data, size_t typesize_data,
  const libxsmm_mhd_element_handler_info* handler_info, libxsmm_mhd_element_handler handler,
  void* minval, void* maxval, int minmax)
{
  int result = EXIT_SUCCESS;
  LIBXSMM_ASSERT(NULL != pitch);
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
              type_data, typesize_data, handler_info, handler, minval, maxval, minmax);
            data = ((const char*)data) + sub_size;
          }
        }
        else result = EXIT_FAILURE;
      }
    }
    else result = EXIT_FAILURE;
  }
  else if (1 == ndims) {
    if (size[0] <= pitch[0]) {
      if ((NULL == handler_info || type_data == handler_info->type) && NULL == handler) { /* fast-path */
        if (size[0] != fwrite(data, ncomponents * typesize_data, size[0], file)) result = EXIT_FAILURE;
      }
      else { /* data-conversion */
        if (0 != minmax) {
          /* determine value-range for scaled data-conversion */
          result = internal_mhd_minmax(data, size[0] * ncomponents, type_data, minval, maxval);
        }
        else {
          const libxsmm_mhd_element_handler handle_element = (NULL == handler
            ? libxsmm_mhd_element_conversion : handler);
          char element[LIBXSMM_MHD_MAX_ELEMSIZE];
          size_t i, j;
          for (i = 0; i < size[0]; ++i) {
            for (j = 0; j < ncomponents; ++j) {
              if (EXIT_SUCCESS == result) {
                const size_t typesize = (NULL == handler_info ? typesize_data
                  : libxsmm_mhd_typesize(handler_info->type));
                result = handle_element(element, handler_info,
                  type_data, data, minval, maxval);
                if (EXIT_SUCCESS == result) {
                  if (1 == fwrite(element, typesize, 1, file)) {
                    data = ((const char*)data) + typesize_data;
                  }
                  else result = EXIT_FAILURE;
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
    else result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_mhd_write(const char filename[],
  const size_t offset[], const size_t size[], const size_t pitch[], size_t ndims,
  size_t ncomponents, libxsmm_mhd_elemtype type_data, const void* data,
  const libxsmm_mhd_element_handler_info* handler_info, libxsmm_mhd_element_handler handler,
  size_t* header_size, const char extension_header[],
  const void* extension, size_t extension_size)
{
  const libxsmm_mhd_elemtype elemtype = (NULL == handler_info ? type_data : handler_info->type);
  const size_t typesize = libxsmm_mhd_typesize(elemtype);
  const char *const elemname = libxsmm_mhd_typename(elemtype, NULL/*ctypename*/);
  FILE *const file = (NULL != filename && 0 != *filename &&
    NULL != size && 0 != ndims && 0 != ncomponents && NULL != data && NULL != elemname && 0 < typesize)
    ? fopen(filename, "wb")
    : NULL;
  int result = EXIT_SUCCESS;
  if (NULL != file) {
    const size_t typesize_data = libxsmm_mhd_typesize(type_data);
    size_t i;
    if (0 < fprintf(file, "NDims = %u\nElementNumberOfChannels = %u\nElementByteOrderMSB = False\nDimSize =",
      (unsigned int)ndims, (unsigned int)ncomponents))
    {
      for (i = 0; i != ndims; ++i) if (0 >= fprintf(file, " %u", (unsigned int)size[i])) {
        result = EXIT_FAILURE;
        break;
      }
    }
    else {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      if (0 < fprintf(file, "\nElementSpacing =")) {
        for (i = 0; i != ndims; ++i) if (0 >= fprintf(file, " 1.0")) {
          result = EXIT_FAILURE;
          break;
        }
      }
      else result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && NULL != extension_header && 0 != *extension_header
      && 0 >= fprintf(file, "\n%s", extension_header))
    {
      result = EXIT_FAILURE;
    }
    /* size of the data, which is silently appended after the regular data section */
    if (EXIT_SUCCESS == result && 0 < extension_size
      && 0 >= fprintf(file, "\nExtensionDataSize = %u", (unsigned int)extension_size))
    {
      result = EXIT_FAILURE;
    }
    /* ElementDataFile must be the last entry before writing the data */
    if (EXIT_SUCCESS == result && 0 < fprintf(file,
      "\nElementType = %s\nElementDataFile = LOCAL\n", elemname))
    {
      size_t pitch1 = 0, size1 = 0;
      const size_t* const shape = (NULL != pitch ? pitch : size);
      const size_t offset1 = libxsmm_offset(offset, shape, ndims, &pitch1);
      const char *const input = ((const char*)data) + offset1 * ncomponents * typesize_data;
      const long file_position = ftell(file); /* determine the header size */
      char minmax[2*(LIBXSMM_MHD_MAX_ELEMSIZE)] = { 0 };
      LIBXSMM_EXPECT(0 == libxsmm_offset(NULL, size, ndims, &size1));
      result = ((0 <= file_position && (offset1 + size1) <= pitch1) ? EXIT_SUCCESS : EXIT_FAILURE);
      if (EXIT_SUCCESS == result && (NULL != handler /* slow-path */
        || (type_data != elemtype && LIBXSMM_MHD_ELEMENT_CONVERSION_DEFAULT == handler_info->hint)))
      { /* conversion needed */
        LIBXSMM_MEMCPY127(minmax, data, typesize_data);
        LIBXSMM_MEMCPY127(minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), data, typesize_data); /* initial condition */
        result = internal_mhd_write(file, input, size, shape, ndims, ncomponents, type_data, typesize_data,
          handler_info, handler, minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), 1/*search min-max*/);
      }
      if (EXIT_SUCCESS == result) {
        if (NULL != header_size) *header_size = file_position;
        assert(file_position == ftell(file)); /* !LIBXSMM_ASSERT */
        result = internal_mhd_write(file, input, size, shape, ndims, ncomponents, type_data, typesize_data,
          handler_info, handler, minmax, minmax + (LIBXSMM_MHD_MAX_ELEMSIZE), 0/*use min-max*/);
      }
    }
    /* append the extension data after the regular data section */
    if (EXIT_SUCCESS == result && 0 < extension_size
      && extension_size != fwrite(extension, 1, extension_size, file))
    {
      result = EXIT_FAILURE;
    }
    /* release file handle */
    if (0 != fclose(file) && EXIT_SUCCESS == result) result = EXIT_FAILURE;
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}
