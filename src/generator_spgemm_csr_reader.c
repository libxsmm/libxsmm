/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_common.h"
#include "generator_spgemm_csr_reader.h"

LIBXSMM_API_INTERN
void libxsmm_sparse_csr_reader( libxsmm_generated_code* io_generated_code,
                                const char*             i_csr_file_in,
                                unsigned int**          o_row_idx,
                                unsigned int**          o_column_idx,
                                double**                o_values,
                                unsigned int*           o_row_count,
                                unsigned int*           o_column_count,
                                unsigned int*           o_element_count ) {
  FILE *l_csr_file_handle;
  const unsigned int l_line_length = 512;
  char l_line[512/*l_line_length*/+1];
  unsigned int l_header_read = 0;
  unsigned int* l_row_idx_id = NULL;
  unsigned int l_i = 0;

  l_csr_file_handle = fopen( i_csr_file_in, "r" );
  if ( l_csr_file_handle == NULL ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_INPUT );
    return;
  }

  while (fgets(l_line, l_line_length, l_csr_file_handle) != NULL) {
    if ( strlen(l_line) == l_line_length ) {
      free(*o_row_idx); free(*o_column_idx); free(*o_values); free(l_row_idx_id);
      *o_row_idx = 0; *o_column_idx = 0; *o_values = 0;
      fclose(l_csr_file_handle); /* close mtx file */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_READ_LEN );
      return;
    }
    /* check if we are still reading comments header */
    if ( l_line[0] == '%' ) {
      continue;
    } else {
      /* if we are the first line after comment header, we allocate our data structures */
      if ( l_header_read == 0 ) {
        if (3 == sscanf(l_line, "%u %u %u", o_row_count, o_column_count, o_element_count) &&
            0 != *o_row_count && 0 != *o_column_count && 0 != *o_element_count)
        {
          /* allocate CSC data-structure matching mtx file */
          /* coverity[tainted_data] */
          *o_column_idx = (unsigned int*) malloc(sizeof(unsigned int) * (*o_element_count));
          /* coverity[tainted_data] */
          *o_row_idx = (unsigned int*) malloc(sizeof(unsigned int) * ((size_t)(*o_row_count) + 1));
          /* coverity[tainted_data] */
          *o_values = (double*) malloc(sizeof(double) * (*o_element_count));
          /* coverity[tainted_data] */
          l_row_idx_id = (unsigned int*) malloc(sizeof(unsigned int) * (*o_row_count));

          /* check if mallocs were successful */
          if ( ( *o_row_idx == NULL )      ||
               ( *o_column_idx == NULL )   ||
               ( *o_values == NULL )       ||
               ( l_row_idx_id == NULL ) ) {
            free(*o_row_idx); free(*o_column_idx); free(*o_values); free(l_row_idx_id);
            *o_row_idx = 0; *o_column_idx = 0; *o_values = 0;
            fclose(l_csr_file_handle); /* close mtx file */
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSC_ALLOC_DATA );
            return;
          }

          /* set everything to zero for init */
          /* coverity[tainted_data] */
          memset(*o_row_idx, 0, sizeof(unsigned int) * ((size_t)(*o_row_count) + 1));
          /* coverity[tainted_data] */
          memset(*o_column_idx, 0, sizeof(unsigned int) * (*o_element_count));
          /* coverity[tainted_data] */
          memset(*o_values, 0, sizeof(double) * (*o_element_count));
          /* coverity[tainted_data] */
          memset(l_row_idx_id, 0, sizeof(unsigned int) * (*o_row_count));

          /* init column idx */
          /* coverity[tainted_data] */
          for ( l_i = 0; l_i <= *o_row_count; ++l_i )
            (*o_row_idx)[l_i] = (*o_element_count);

          /* init */
          (*o_row_idx)[0] = 0;
          l_i = 0;
          l_header_read = 1;
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_READ_DESC );
          fclose( l_csr_file_handle ); /* close mtx file */
          return;
        }
      /* now we read the actual content */
      } else {
        unsigned int l_row = 0, l_column = 0;
        double l_value = 0;
        /* read a line of content */
        if ( sscanf(l_line, "%u %u %lf", &l_row, &l_column, &l_value) != 3 ) {
          free(*o_row_idx); free(*o_column_idx); free(*o_values); free(l_row_idx_id);
          *o_row_idx = 0; *o_column_idx = 0; *o_values = 0;
          fclose(l_csr_file_handle); /* close mtx file */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_READ_ELEMS );
          return;
        }
        /* adjust numbers to zero termination */
        LIBXSMM_ASSERT(0 != l_row && 0 != l_column);
        l_row--; l_column--;
        /* add these values to row and value structure */
        (*o_column_idx)[l_i] = l_column;
        (*o_values)[l_i] = l_value;
        l_i++;
        /* handle columns, set id to own for this column, yeah we need to handle empty columns */
        /* coverity[tainted_data] */
        l_row_idx_id[l_row] = 1;
        (*o_row_idx)[l_row+1] = l_i;
      }
    }
  }

  /* close mtx file */
  fclose( l_csr_file_handle );

  /* check if we read a file which was consistent */
  if ( l_i != (*o_element_count) ) {
    free(*o_row_idx); free(*o_column_idx); free(*o_values); free(l_row_idx_id);
    *o_row_idx = 0; *o_column_idx = 0; *o_values = 0;
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_LEN );
    return;
  }

  if ( l_row_idx_id != NULL ) {
    /* let's handle empty rows */
    for ( l_i = 0; l_i < (*o_row_count); l_i++) {
      if ( l_row_idx_id[l_i] == 0 ) {
        (*o_row_idx)[l_i+1] = (*o_row_idx)[l_i];
      }
    }

    /* free helper data structure */
    free( l_row_idx_id );
  }
}

