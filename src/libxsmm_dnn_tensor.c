/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include "libxsmm_main.h"
#include "libxsmm_dnn_tensor.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_link_tensor(const libxsmm_dnn_tensor_datalayout* layout, const void* data, libxsmm_dnn_err_t* status)
{
  return libxsmm_dnn_link_qtensor(layout, data, 0, status);
}


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_link_qtensor(const libxsmm_dnn_tensor_datalayout* layout, const void* data, const unsigned char scf, libxsmm_dnn_err_t* status)
{
  /* zero entire content; not only safer but also sets data and code pointers to NULL */
  libxsmm_dnn_tensor* tensor = (libxsmm_dnn_tensor*)calloc(1, sizeof(libxsmm_dnn_tensor));
  *status = LIBXSMM_DNN_SUCCESS;

  if (layout != 0 && tensor != 0 && data != 0) {
    tensor->layout = libxsmm_dnn_duplicate_tensor_datalayout(layout, status);
    tensor->data = (void*)data;
    tensor->scf = scf;
    /* when layout copy failed, free layout */
    if (*status != LIBXSMM_DNN_SUCCESS) {
      libxsmm_dnn_destroy_tensor_datalayout(tensor->layout);
    }
  } else {
    *status = LIBXSMM_DNN_ERR_CREATE_TENSOR;
  }

  if (*status != LIBXSMM_DNN_SUCCESS) {
    free((libxsmm_dnn_tensor*)tensor);
    tensor = 0;
  }

  return tensor;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_duplicate_tensor_datalayout(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* dst_layout;

  *status = LIBXSMM_DNN_SUCCESS;
  dst_layout = 0;

  if (layout != 0 && layout->num_dims != 0) {
    unsigned int dim = 0;

    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    dst_layout = (libxsmm_dnn_tensor_datalayout*)calloc(1, sizeof(libxsmm_dnn_tensor_datalayout));
    if (0 != dst_layout) {
      dst_layout->dim_type = (libxsmm_dnn_tensor_dimtype*)malloc(layout->num_dims * sizeof(libxsmm_dnn_tensor_dimtype));
      dst_layout->dim_size = (unsigned int*)malloc(layout->num_dims * sizeof(unsigned int));
      dst_layout->num_dims = layout->num_dims;
      dst_layout->format = layout->format;
      dst_layout->datatype = layout->datatype;
      dst_layout->tensor_type = layout->tensor_type;
      if (0 != dst_layout->dim_type && 0 != dst_layout->dim_size) {
        for (dim = 0; dim < layout->num_dims; ++dim) {
          dst_layout->dim_type[dim] = layout->dim_type[dim];
          dst_layout->dim_size[dim] = layout->dim_size[dim];
        }
      } else {
        *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT;
      }
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_LAYOUT;
  }

  return dst_layout;
}


LIBXSMM_API unsigned int libxsmm_dnn_compare_tensor_datalayout(const libxsmm_dnn_tensor_datalayout* layout_a, const libxsmm_dnn_tensor_datalayout* layout_b, libxsmm_dnn_err_t* status) {
  unsigned int result = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (layout_a != 0 && layout_b != 0) {
    unsigned int dim = 0;

    if (layout_a->num_dims      != layout_b->num_dims)      { result = 1; }
    if (layout_a->format        != layout_b->format)        { result = 1; }
    if (layout_a->datatype      != layout_b->datatype)      { result = 1; }

    if (result == 0) {
      for ( dim = 0; dim < layout_a->num_dims; ++dim ) {
        if ( layout_a->dim_type[dim] != layout_b->dim_type[dim] ) { result = 1; }
        if ( layout_a->dim_size[dim] != layout_b->dim_size[dim] ) { result = 1; }
      }
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_LAYOUT;
    result = 100;
  }

  return result;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_tensor_datalayout(libxsmm_dnn_tensor_datalayout* layout) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != layout) {
    free(layout->dim_type);
    free(layout->dim_size);
    free(layout);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_LAYOUT;
  }

  return status;
}


LIBXSMM_API unsigned int libxsmm_dnn_get_tensor_size(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status) {
  unsigned int size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != layout) {
    unsigned int dim = 0;
    size = (unsigned int)libxsmm_dnn_typesize(layout->datatype);
    for (dim = 0; dim < layout->num_dims; ++dim) {
      size *= layout->dim_size[dim];
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_LAYOUT;
  }

  return size;
}


LIBXSMM_API unsigned int libxsmm_dnn_get_tensor_elements(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status) {
  unsigned int elements = 1;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != layout) {
    unsigned int dim = 0;
    for ( dim = 0; dim < layout->num_dims; ++dim ) {
      elements *= layout->dim_size[dim];
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_LAYOUT;
    elements = 0;
  }

  return elements;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_tensor* tensor, const void* data) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if ((0 != tensor) && (0 != data)) {
    if (0 != tensor->layout) {
      if (0 < tensor->layout->num_dims) {
        tensor->data = (void*)data;
      } else {
        status = LIBXSMM_DNN_ERR_INVALID_LAYOUT;
      }
    } else {
      status = LIBXSMM_DNN_ERR_INVALID_LAYOUT;
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}


LIBXSMM_API void* libxsmm_dnn_get_tensor_data_ptr(const libxsmm_dnn_tensor* tensor, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != tensor) {
    return tensor->data;
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }

  return 0;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_get_tensor_datalayout(const libxsmm_dnn_tensor* tensor, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* dst_layout = NULL;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != tensor) {
    dst_layout = libxsmm_dnn_duplicate_tensor_datalayout( tensor->layout, status );
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }

  return dst_layout;
}


LIBXSMM_API unsigned char libxsmm_dnn_get_qtensor_scf(const libxsmm_dnn_tensor* tensor, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != tensor) {
    return tensor->scf;
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }

  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_set_qtensor_scf(libxsmm_dnn_tensor* tensor, const unsigned char scf)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != tensor) {
    tensor->scf = scf;
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_tensor(const libxsmm_dnn_tensor* tensor)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != tensor) { /* it is not an error attempting to destroy a NULL-handle */
    /* free layout information stored in tensor */
    if (0 != tensor->layout) {
      libxsmm_dnn_destroy_tensor_datalayout( (libxsmm_dnn_tensor_datalayout*)tensor->layout );
    }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_tensor*)tensor);
  }
#if 0 /* releasing a NULL-buffer should be not an error (similar to freeing a NULL pointer) */
  else {
    status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }
#endif
  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyin_tensor(const libxsmm_dnn_tensor* tensor, const void* data, const libxsmm_dnn_tensor_format in_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* @TODO check for valid combination */

  if (0 != tensor) {
    switch (tensor->layout->tensor_type) {
      case LIBXSMM_DNN_REGULAR_INPUT:
      case LIBXSMM_DNN_GRADIENT_INPUT:
      case LIBXSMM_DNN_REGULAR_OUTPUT:
      case LIBXSMM_DNN_GRADIENT_OUTPUT:
      case LIBXSMM_DNN_INPUT:
      case LIBXSMM_DNN_OUTPUT:
      case LIBXSMM_DNN_ACTIVATION: {
                                     switch (in_format) {
                                       case LIBXSMM_DNN_TENSOR_FORMAT_NCHW: {
                                                                              if ( (tensor->layout->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
                                                                                switch (tensor->layout->datatype) {
                                                                                  case LIBXSMM_DNN_DATATYPE_F32: {
                                                                                                                   typedef float element_type;
#include "template/libxsmm_dnn_tensor_buffer_copy_in_nchw.tpl.c"
                                                                                                                 } break;
                                                                                  case LIBXSMM_DNN_DATATYPE_BF16: {
                                                                                                                   typedef libxsmm_bfloat16 element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_buffer_copy_in_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                                 } break;
                                                                                  case LIBXSMM_DNN_DATATYPE_I32: {
                                                                                                                   typedef int element_type;
#include "template/libxsmm_dnn_tensor_buffer_copy_in_nchw.tpl.c"
                                                                                                                 } break;
                                                                                  case LIBXSMM_DNN_DATATYPE_I16: {
                                                                                                                   typedef short  element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_buffer_copy_in_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                                 } break;
                                                                                  case LIBXSMM_DNN_DATATYPE_I8: {
                                                                                                                  typedef unsigned char element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_buffer_copy_in_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                                } break;
                                                                                  default: {
                                                                                             status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
                                                                                           }
                                                                                }
                                                                              } else {
                                                                                status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
                                                                              }
                                                                            } break;
                                       default: {
                                                  status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
                                                }
                                     }
                                   } break;
      case LIBXSMM_DNN_REGULAR_FILTER:
      case LIBXSMM_DNN_GRADIENT_FILTER:
      case LIBXSMM_DNN_FILTER: {
                                 switch (in_format) {
                                   case LIBXSMM_DNN_TENSOR_FORMAT_KCRS: {
                                                                          if ( (tensor->layout->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
                                                                            switch (tensor->layout->datatype) {
                                                                              case LIBXSMM_DNN_DATATYPE_F32: {
                                                                                                               typedef float element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_in_kcrs.tpl.c"
                                                                                                             } break;
                                                                              case LIBXSMM_DNN_DATATYPE_BF16: {
                                                                                                               typedef libxsmm_bfloat16 element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_in_kcrs.tpl.c"
                                                                                                             } break;
                                                                              case LIBXSMM_DNN_DATATYPE_I16: {
                                                                                                               typedef short element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_in_kcrs.tpl.c"
                                                                                                             } break;
                                                                              case LIBXSMM_DNN_DATATYPE_I8: {
                                                                                                              typedef char element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_in_kcrs.tpl.c"
                                                                                                            } break;
                                                                              default: {
                                                                                         status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
                                                                                       }
                                                                            }
                                                                          } else {
                                                                            status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
                                                                          }
                                                                        } break;
                                   default: {
                                              status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
                                            }
                                 }
                               } break;
      case LIBXSMM_DNN_REGULAR_CHANNEL_BIAS:
      case LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS:
      case LIBXSMM_DNN_CHANNEL_BIAS:
      case LIBXSMM_DNN_REGULAR_CHANNEL_BETA:
      case LIBXSMM_DNN_GRADIENT_CHANNEL_BETA:
      case LIBXSMM_DNN_CHANNEL_BETA:
      case LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA:
      case LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA:
      case LIBXSMM_DNN_CHANNEL_GAMMA:
      case LIBXSMM_DNN_CHANNEL_EXPECTVAL:
      case LIBXSMM_DNN_CHANNEL_RCPSTDDEV:
      case LIBXSMM_DNN_CHANNEL_VARIANCE:
      case LIBXSMM_DNN_CHANNEL_SCALAR: {
                               switch (in_format) {
                                 case LIBXSMM_DNN_TENSOR_FORMAT_NCHW: {
                                                                        if ( (tensor->layout->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
                                                                          switch (tensor->layout->datatype) {
                                                                            case LIBXSMM_DNN_DATATYPE_F32: {
                                                                                                             typedef float element_type;
#include "template/libxsmm_dnn_tensor_bias_copy_in_nchw.tpl.c"
                                                                                                           } break;
                                                                            case LIBXSMM_DNN_DATATYPE_BF16: {
                                                                                                             typedef libxsmm_bfloat16 element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_bias_copy_in_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                           } break;
                                                                            case LIBXSMM_DNN_DATATYPE_I16: {
                                                                                                             typedef short element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_bias_copy_in_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                           } break;
                                                                            case LIBXSMM_DNN_DATATYPE_I8: {
                                                                                                            typedef char element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_bias_copy_in_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                          } break;
                                                                            default: {
                                                                                       status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
                                                                                     }
                                                                          }
                                                                        } else {
                                                                          status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
                                                                        }
                                                                      } break;
                                 default: {
                                            status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
                                          }
                               }
                             } break;
      default: {
                 status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
               }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_zero_tensor(const libxsmm_dnn_tensor* tensor)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != tensor) {
    const size_t size = libxsmm_dnn_get_tensor_elements( tensor->layout, &status );
    size_t i;
    /* use for-loops to potentially leverage NUMA in the future */
    switch (tensor->layout->datatype) {
      case LIBXSMM_DNN_DATATYPE_F32: {
                                       float* fp32_data = (float*)tensor->data;
                                       for (i = 0; i < size; ++i) fp32_data[i] = 0.0f;
                                     } break;
      case LIBXSMM_DNN_DATATYPE_BF16: {
                                       libxsmm_bfloat16* bfp16_data = (libxsmm_bfloat16*)tensor->data;
                                       for (i = 0; i < size; ++i) bfp16_data[i] = 0;
                                     } break;
      case LIBXSMM_DNN_DATATYPE_I32: {
                                       int* int32_data = (int*)tensor->data;
                                       for (i = 0; i < size; ++i) int32_data[i] = 0;
                                     } break;
      case LIBXSMM_DNN_DATATYPE_I16: {
                                       short* int16_data = (short*)tensor->data;
                                       for (i = 0; i < size; ++i) int16_data[i] = 0;
                                     } break;
      case LIBXSMM_DNN_DATATYPE_I8: {
                                      char* int8_data = (char*)tensor->data;
                                      for (i = 0; i < size; ++i) int8_data[i] = 0;
                                    } break;
      default: {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_tensor(const libxsmm_dnn_tensor* tensor, void* data, const libxsmm_dnn_tensor_format out_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* @TODO check for valid combination */

  if (0 != tensor) {
    switch (tensor->layout->tensor_type) {
      case LIBXSMM_DNN_REGULAR_INPUT:
      case LIBXSMM_DNN_GRADIENT_INPUT:
      case LIBXSMM_DNN_REGULAR_OUTPUT:
      case LIBXSMM_DNN_GRADIENT_OUTPUT:
      case LIBXSMM_DNN_INPUT:
      case LIBXSMM_DNN_OUTPUT:
      case LIBXSMM_DNN_ACTIVATION: {
                                     switch (out_format) {
                                       case LIBXSMM_DNN_TENSOR_FORMAT_NCHW: {
                                                                              if ( (tensor->layout->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
                                                                                switch (tensor->layout->datatype) {
                                                                                  case LIBXSMM_DNN_DATATYPE_F32: {
                                                                                                                   typedef float element_type;
#include "template/libxsmm_dnn_tensor_buffer_copy_out_nchw.tpl.c"
                                                                                                                 } break;
                                                                                  case LIBXSMM_DNN_DATATYPE_BF16: {
                                                                                                                   typedef libxsmm_bfloat16 element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_buffer_copy_out_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                                 } break;
                                                                                  case LIBXSMM_DNN_DATATYPE_I32: {
                                                                                                                   typedef int element_type;
#include "template/libxsmm_dnn_tensor_buffer_copy_out_nchw.tpl.c"
                                                                                                                 } break;
                                                                                  case LIBXSMM_DNN_DATATYPE_I16: {
                                                                                                                   typedef short element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_buffer_copy_out_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                                 } break;
                                                                                  case LIBXSMM_DNN_DATATYPE_I8: {
                                                                                                                  typedef unsigned char element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_buffer_copy_out_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                                } break;
                                                                                  default: {
                                                                                             status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
                                                                                           }
                                                                                }
                                                                              } else {
                                                                                status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
                                                                              }
                                                                            } break;
                                       default: {
                                                  status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
                                                }
                                     }
                                   } break;
      case LIBXSMM_DNN_REGULAR_FILTER:
      case LIBXSMM_DNN_GRADIENT_FILTER:
      case LIBXSMM_DNN_FILTER: {
                                 switch (out_format) {
                                   case LIBXSMM_DNN_TENSOR_FORMAT_KCRS: {
                                                                          if ( (tensor->layout->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
                                                                            switch (tensor->layout->datatype) {
                                                                              case LIBXSMM_DNN_DATATYPE_F32: {
                                                                                                               typedef float element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_out_kcrs.tpl.c"
                                                                                                             } break;

                                                                              case LIBXSMM_DNN_DATATYPE_BF16: {
                                                                                                               typedef libxsmm_bfloat16 element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_out_kcrs.tpl.c"
                                                                                                             } break;
                                                                                   case LIBXSMM_DNN_DATATYPE_I32: {
                                                                                                                   typedef int element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_out_kcrs.tpl.c"
                                                                                                                 } break;
                                                                                   case LIBXSMM_DNN_DATATYPE_I16: {
                                                                                                               typedef short  element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_out_kcrs.tpl.c"
                                                                                                             } break;
                                                                              case LIBXSMM_DNN_DATATYPE_I8: {
                                                                                                              typedef char element_type;
#include "template/libxsmm_dnn_tensor_filter_copy_out_kcrs.tpl.c"
                                                                                                            } break;
                                                                              default: {
                                                                                         status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
                                                                                       }
                                                                            }
                                                                          } else {
                                                                            status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
                                                                          }
                                                                        } break;
                                   default: {
                                              status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
                                            }
                                 }
                               } break;
      case LIBXSMM_DNN_REGULAR_CHANNEL_BIAS:
      case LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS:
      case LIBXSMM_DNN_CHANNEL_BIAS:
      case LIBXSMM_DNN_REGULAR_CHANNEL_BETA:
      case LIBXSMM_DNN_GRADIENT_CHANNEL_BETA:
      case LIBXSMM_DNN_CHANNEL_BETA:
      case LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA:
      case LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA:
      case LIBXSMM_DNN_CHANNEL_GAMMA:
      case LIBXSMM_DNN_CHANNEL_EXPECTVAL:
      case LIBXSMM_DNN_CHANNEL_RCPSTDDEV:
      case LIBXSMM_DNN_CHANNEL_VARIANCE:
      case LIBXSMM_DNN_CHANNEL_SCALAR: {
                               switch (out_format) {
                                 case LIBXSMM_DNN_TENSOR_FORMAT_NCHW: {
                                                                        if ( (tensor->layout->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
                                                                          switch (tensor->layout->datatype) {
                                                                            case LIBXSMM_DNN_DATATYPE_F32: {
                                                                                                             typedef float element_type;
#include "template/libxsmm_dnn_tensor_bias_copy_out_nchw.tpl.c"
                                                                                                           } break;
                                                                            case LIBXSMM_DNN_DATATYPE_BF16: {
                                                                                                             typedef libxsmm_bfloat16 element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_bias_copy_out_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                           } break;
                                                                            case LIBXSMM_DNN_DATATYPE_I16: {
                                                                                                             typedef short element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_bias_copy_out_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                           } break;
                                                                            case LIBXSMM_DNN_DATATYPE_I8: {
                                                                                                            typedef char element_type;
#define LIBXSMM_DNN_COPY_LOW_PRECISION
#include "template/libxsmm_dnn_tensor_bias_copy_out_nchw.tpl.c"
#undef LIBXSMM_DNN_COPY_LOW_PRECISION
                                                                                                          } break;
                                                                            default: {
                                                                                       status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
                                                                                     }
                                                                          }
                                                                        } else {
                                                                          status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
                                                                        }
                                                                      } break;
                                 default: {
                                            status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
                                          }
                               }
                             } break;
      default: {
                 status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
               }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}

