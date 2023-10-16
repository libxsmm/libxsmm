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
#ifndef LIBXSMM_UTILS_MHD_H
#define LIBXSMM_UTILS_MHD_H

#include "../libxsmm_typedefs.h"


/** Denotes the element/pixel type of an image/channel. */
typedef enum libxsmm_mhd_elemtype {
  LIBXSMM_MHD_ELEMTYPE_F64  = LIBXSMM_DATATYPE_F64,         /* MET_DOUBLE */
  LIBXSMM_MHD_ELEMTYPE_F32  = LIBXSMM_DATATYPE_F32,         /* MET_FLOAT */
  LIBXSMM_MHD_ELEMTYPE_F16  = LIBXSMM_DATATYPE_F16,         /* MET_HALF */
  LIBXSMM_MHD_ELEMTYPE_BF16 = LIBXSMM_DATATYPE_BF16,        /* MET_BFLOAT */
  LIBXSMM_MHD_ELEMTYPE_BF8  = LIBXSMM_DATATYPE_BF8,         /* MET_BFLOAT8 */
  LIBXSMM_MHD_ELEMTYPE_I64  = LIBXSMM_DATATYPE_I64,         /* MET_LONG */
  LIBXSMM_MHD_ELEMTYPE_I32  = LIBXSMM_DATATYPE_I32,         /* MET_INT */
  LIBXSMM_MHD_ELEMTYPE_I16  = LIBXSMM_DATATYPE_I16,         /* MET_SHORT */
  LIBXSMM_MHD_ELEMTYPE_I8   = LIBXSMM_DATATYPE_I8,          /* MET_CHAR */
  LIBXSMM_MHD_ELEMTYPE_U64  = LIBXSMM_DATATYPE_UNSUPPORTED, /* MET_ULONG */
  LIBXSMM_MHD_ELEMTYPE_U32, /* MET_UINT */
  LIBXSMM_MHD_ELEMTYPE_U16, /* MET_USHORT */
  LIBXSMM_MHD_ELEMTYPE_U8,  /* MET_UCHAR */
  LIBXSMM_MHD_ELEMTYPE_UNKNOWN
} libxsmm_mhd_elemtype;


/**
 * Function type used for custom data-handler or element conversion.
 * The value-range (src_min, src_max) may be used to scale values
 * in case of a type-conversion.
 */
LIBXSMM_EXTERN_C typedef int (*libxsmm_mhd_element_handler)(
  void* dst, libxsmm_mhd_elemtype dst_type, libxsmm_mhd_elemtype src_type,
  const void* src, const void* src_min, const void* src_max);

/**
 * Predefined function to perform element data conversion.
 * Scales source-values in case of non-NULL src_min and src_max,
 * or otherwise clamps to the destination-type.
 */
LIBXSMM_API int libxsmm_mhd_element_conversion(
  void* dst, libxsmm_mhd_elemtype dst_type, libxsmm_mhd_elemtype src_type,
  const void* src, const void* src_min, const void* src_max);

/**
 * Predefined function to check a buffer against file content.
 * In case of different types, libxsmm_mhd_element_conversion
 * is performed to compare values using the source-type.
 */
LIBXSMM_API int libxsmm_mhd_element_comparison(
  void* dst, libxsmm_mhd_elemtype dst_type, libxsmm_mhd_elemtype src_type,
  const void* src, const void* src_min, const void* src_max);


/** Returns the name and size of the element type; result may be NULL/0 in case of an unknown type. */
LIBXSMM_API const char* libxsmm_mhd_typename(libxsmm_mhd_elemtype type, size_t* typesize, const char** ctypename);

/** Returns the type of the element for a given type-name. */
LIBXSMM_API libxsmm_mhd_elemtype libxsmm_mhd_typeinfo(const char elemname[]);


/**
 * Parse the header of an MHD-file. The header can be part of the data file (local),
 * or separately stored (header: MHD, data MHA or RAW).
 */
LIBXSMM_API int libxsmm_mhd_read_header(
  /* Filename referring to the header-file (may also contain the data). */
  const char header_filename[],
  /* Maximum length of path/file name. */
  size_t filename_max_length,
  /* Filename containing the data (may be the same as the header-file). */
  char filename[],
  /* Yields the maximum/possible number of dimensions on input,
   * and the actual number of dimensions on output. */
  size_t* ndims,
  /* Image extents ("ndims" number of entries). */
  size_t size[],
  /* Number of interleaved image channels. */
  size_t* ncomponents,
  /* Type of the image elements (pixel type). */
  libxsmm_mhd_elemtype* type,
  /* Size of the header in bytes; may be used to skip the header,
   * when reading content; can be a NULL-argument (optional). */
  size_t* header_size,
  /* Size (in Bytes) of an user-defined extended data record;
   * can be a NULL-argument (optional). */
  size_t* extension_size);


/**
 * Loads the data file, and optionally allows data conversion.
 * Conversion is performed such that values are clamped to fit
 * into the destination.
 */
LIBXSMM_API int libxsmm_mhd_read(
  /* Filename referring to the data. */
  const char filename[],
  /* Offset within pitched buffer (NULL: no offset). */
  const size_t offset[],
  /* Image dimensions (extents). */
  const size_t size[],
  /* Leading buffer dimensions (NULL: same as size). */
  const size_t pitch[],
  /* Dimensionality (number of entries in size). */
  size_t ndims,
  /* Number of interleaved image channels. */
  size_t ncomponents,
  /* Used to skip the header, and to only read the data. */
  size_t header_size,
  /* Data element type as stored (pixel type). */
  libxsmm_mhd_elemtype type_stored,
  /* Storage type (data conversion, optional). */
  const libxsmm_mhd_elemtype* type_data,
  /* Buffer where the data is read into. */
  void* data,
  /**
   * Optional callback executed per entry when reading the data.
   * May assign the value to the left-most argument, but also
   * allows to only compare with present data. Can be used to
   * avoid allocating an actual destination.
   */
  libxsmm_mhd_element_handler handle_element,
  /* Post-content data (extension, optional). */
  char extension[],
  /* Size of the extension; can be zero. */
  size_t extension_size);


/**
 * Save a file using an extended data format, which is compatible with the Meta Image Format (MHD).
 * The file is suitable for visual inspection using, e.g., ITK-SNAP or ParaView.
 */
LIBXSMM_API int libxsmm_mhd_write(const char filename[],
  /* Offset within pitched buffer (NULL: no offset). */
  const size_t offset[],
  /* Image dimensions (extents). */
  const size_t size[],
  /* Leading buffer dimensions (NULL: same as size). */
  const size_t pitch[],
  /* Dimensionality, i.e., number of entries in data_size/size. */
  size_t ndims,
  /* Number of pixel components. */
  size_t ncomponents,
  /* Type (input). */
  libxsmm_mhd_elemtype type_data,
  /* Type (data conversion, optional). */
  const libxsmm_mhd_elemtype* type,
  /* Raw data to be saved. */
  const void* data,
  /* Size of the header; can be a NULL-argument (optional). */
  size_t* header_size,
  /* Extension header data; can be NULL. */
  const char extension_header[],
  /* Extension data stream; can be NULL. */
  const void* extension,
  /* Extension data size; can be NULL. */
  size_t extension_size);

#endif /*LIBXSMM_UTILS_MHD_H*/
