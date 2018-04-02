/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#if defined(__EDGE_EXECUTE_F32__)
#define REALTYPE float
#else
#define REALTYPE double
#endif

typedef struct edge_mat_desc {
  unsigned int row_count;
  unsigned int col_count;
  unsigned int num_elements;
} edge_mat_desc;

static void libxsmm_sparse_csr_reader( const char*    i_csr_file_in,
                                unsigned int**        o_row_idx,
                                unsigned int**        o_column_idx,
                                REALTYPE**            o_values,
                                unsigned int*         o_row_count,
                                unsigned int*         o_column_count,
                                unsigned int*         o_element_count ) {
  FILE *l_csr_file_handle;
  const unsigned int l_line_length = 512;
  char l_line[512/*l_line_length*/+1];
  unsigned int l_header_read = 0;
  unsigned int* l_row_idx_id = NULL;
  unsigned int l_i = 0;

  l_csr_file_handle = fopen( i_csr_file_in, "r" );
  if ( l_csr_file_handle == NULL ) {
    fprintf( stderr, "cannot open CSR file!\n" );
    return;
  }

  while (fgets(l_line, l_line_length, l_csr_file_handle) != NULL) {
    if ( strlen(l_line) == l_line_length ) {
      fprintf( stderr, "could not read file length!\n" );
      return;
    }
    /* check if we are still reading comments header */
    if ( l_line[0] == '%' ) {
      continue;
    } else {
      /* if we are the first line after comment header, we allocate our data structures */
      if ( l_header_read == 0 ) {
        if ( sscanf(l_line, "%u %u %u", o_row_count, o_column_count, o_element_count) == 3 ) {
          /* allocate CSC datastructure matching mtx file */
          *o_column_idx = (unsigned int*) malloc(sizeof(unsigned int) * (*o_element_count));
          *o_row_idx = (unsigned int*) malloc(sizeof(unsigned int) * (*o_row_count + 1));
          *o_values = (REALTYPE*) malloc(sizeof(double) * (*o_element_count));
          l_row_idx_id = (unsigned int*) malloc(sizeof(unsigned int) * (*o_row_count));

          /* check if mallocs were successful */
          if ( ( *o_row_idx == NULL )      ||
               ( *o_column_idx == NULL )   ||
               ( *o_values == NULL )       ||
               ( l_row_idx_id == NULL )    ) {
            fprintf( stderr, "could not allocate sp data!\n" );
            return;
          }

          /* set everything to zero for init */
          memset(*o_row_idx, 0, sizeof(unsigned int)*(*o_row_count + 1));
          memset(*o_column_idx, 0, sizeof(unsigned int)*(*o_element_count));
          memset(*o_values, 0, sizeof(double)*(*o_element_count));
          memset(l_row_idx_id, 0, sizeof(unsigned int)*(*o_row_count));

          /* init column idx */
          for ( l_i = 0; l_i < (*o_row_count + 1); l_i++)
            (*o_row_idx)[l_i] = (*o_element_count);

          /* init */
          (*o_row_idx)[0] = 0;
          l_i = 0;
          l_header_read = 1;
        } else {
          fprintf( stderr, "could not csr description!\n" );
          return;
        }
      /* now we read the actual content */
      } else {
        unsigned int l_row, l_column;
        REALTYPE l_value;
        /* read a line of content */
#if defined(__EDGE_EXECUTE_F32__)
        if ( sscanf(l_line, "%u %u %f", &l_row, &l_column, &l_value) != 3 ) {
          fprintf( stderr, "could not read element!\n" );
          return;
        }
#else
        if ( sscanf(l_line, "%u %u %lf", &l_row, &l_column, &l_value) != 3 ) {
          fprintf( stderr, "could not read element!\n" );
          return;
        }
#endif
        /* adjust numbers to zero termination */
        l_row--;
        l_column--;
        /* add these values to row and value structure */
        (*o_column_idx)[l_i] = l_column;
        (*o_values)[l_i] = l_value;
        l_i++;
        /* handle columns, set id to own for this column, yeah we need to handle empty columns */
        l_row_idx_id[l_row] = 1;
        (*o_row_idx)[l_row+1] = l_i;
      }
    }
  }

  /* close mtx file */
  fclose( l_csr_file_handle );

  /* check if we read a file which was consistent */
  if ( l_i != (*o_element_count) ) {
    fprintf( stderr, "we were not able to read all elements!\n" );
    return;
  }

  /* let's handle empty rows */
  for ( l_i = 0; l_i < (*o_row_count); l_i++) {
    if ( l_row_idx_id[l_i] == 0 ) {
      (*o_row_idx)[l_i+1] = (*o_row_idx)[l_i];
    }
  }

  /* free helper data structure */
  if ( l_row_idx_id != NULL ) {
    free( l_row_idx_id );
  }
}

static void libxsmm_sparse_csc_reader( const char*    i_csc_file_in,
                                unsigned int**        o_column_idx,
                                unsigned int**        o_row_idx,
                                REALTYPE**            o_values,
                                unsigned int*         o_row_count,
                                unsigned int*         o_column_count,
                                unsigned int*         o_element_count ) {
  FILE *l_csc_file_handle;
  const unsigned int l_line_length = 512;
  char l_line[512/*l_line_length*/+1];
  unsigned int l_header_read = 0;
  unsigned int* l_column_idx_id = NULL;
  unsigned int l_i = 0;

  l_csc_file_handle = fopen( i_csc_file_in, "r" );
  if ( l_csc_file_handle == NULL ) {
    fprintf( stderr, "cannot open CSC file!\n" );
    return;
  }

  while (fgets(l_line, l_line_length, l_csc_file_handle) != NULL) {
    if ( strlen(l_line) == l_line_length ) {
      fprintf( stderr, "could not read file length!\n" );
      return;
    }
    /* check if we are still reading comments header */
    if ( l_line[0] == '%' ) {
      continue;
    } else {
      /* if we are the first line after comment header, we allocate our data structures */
      if ( l_header_read == 0 ) {
        if ( sscanf(l_line, "%u %u %u", o_row_count, o_column_count, o_element_count) == 3 ) {
          /* allocate CSC datastructure matching mtx file */
          *o_row_idx = (unsigned int*) malloc(sizeof(unsigned int) * (*o_element_count));
          *o_column_idx = (unsigned int*) malloc(sizeof(unsigned int) * (*o_column_count + 1));
          *o_values = (REALTYPE*) malloc(sizeof(double) * (*o_element_count));
          l_column_idx_id = (unsigned int*) malloc(sizeof(unsigned int) * (*o_column_count));

          /* check if mallocs were successful */
          if ( ( *o_row_idx == NULL )      ||
               ( *o_column_idx == NULL )   ||
               ( *o_values == NULL )       ||
               ( l_column_idx_id == NULL )    ) {
            fprintf( stderr, "could not allocate sp data!\n" );
            return;
          }

          /* set everything to zero for init */
          memset(*o_column_idx, 0, sizeof(unsigned int)*(*o_column_count + 1));
          memset(*o_row_idx, 0, sizeof(unsigned int)*(*o_element_count));
          memset(*o_values, 0, sizeof(double)*(*o_element_count));
          memset(l_column_idx_id, 0, sizeof(unsigned int)*(*o_column_count));

          /* init column idx */
          for ( l_i = 0; l_i < (*o_column_count + 1); l_i++)
            (*o_column_idx)[l_i] = (*o_element_count);

          /* init */
          (*o_column_idx)[0] = 0;
          l_i = 0;
          l_header_read = 1;
        } else {
          fprintf( stderr, "could not csr description!\n" );
          return;
        }
      /* now we read the actual content */
      } else {
        unsigned int l_row, l_column;
        REALTYPE l_value;
        /* read a line of content */
#if defined(__EDGE_EXECUTE_F32__)
        if ( sscanf(l_line, "%u %u %f", &l_row, &l_column, &l_value) != 3 ) {
          fprintf( stderr, "could not read element!\n" );
          return;
        }
#else
        if ( sscanf(l_line, "%u %u %lf", &l_row, &l_column, &l_value) != 3 ) {
          fprintf( stderr, "could not read element!\n" );
          return;
        }
#endif
        /* adjust numbers to zero termination */
        l_row--;
        l_column--;
        /* add these values to row and value structure */
        (*o_row_idx)[l_i] = l_row;
        (*o_values)[l_i] = l_value;
        l_i++;
        /* handle columns, set id to own for this column, yeah we need to handle empty columns */
        l_column_idx_id[l_column] = 1;
        (*o_column_idx)[l_column+1] = l_i;
      }
    }
  }

  /* close mtx file */
  fclose( l_csc_file_handle );

  /* check if we read a file which was consistent */
  if ( l_i != (*o_element_count) ) {
    fprintf( stderr, "we were not able to read all elements!\n" );
    return;
  }

  /* let's handle empty rows */
  for ( l_i = 0; l_i < (*o_column_count); l_i++) {
    if ( l_column_idx_id[l_i] == 0 ) {
      (*o_column_idx)[l_i+1] = (*o_column_idx)[l_i];
    }
  }

  /* free helper data structure */
  if ( l_column_idx_id != NULL ) {
    free( l_column_idx_id );
  }
}

static edge_mat_desc libxsmm_sparse_csr_reader_desc( const char*    i_csr_file_in ) {
  FILE *l_csr_file_handle;
  const unsigned int l_line_length = 512;
  char l_line[512/*l_line_length*/+1];
  unsigned int l_header_read = 0;
  unsigned int l_row_count = 0;
  unsigned int l_col_count = 0;
  unsigned int l_num_elements = 0;
  edge_mat_desc desc;

  desc.row_count = 0;
  desc.col_count = 0;
  desc.num_elements = 0;

  l_csr_file_handle = fopen( i_csr_file_in, "r" );
  if ( l_csr_file_handle == NULL ) {
    fprintf( stderr, "cannot open CSR file!\n" );
    return desc;
  }

  while (fgets(l_line, l_line_length, l_csr_file_handle) != NULL) {
    if ( strlen(l_line) == l_line_length ) {
      fprintf( stderr, "could not read file length!\n" );
      return desc;
    }
    /* check if we are still reading comments header */
    if ( l_line[0] == '%' ) {
      continue;
    } else {
      /* if we are the first line after comment header, we allocate our data structures */
      if ( l_header_read == 0 ) {
        if ( sscanf(l_line, "%u %u %u", &l_row_count, &l_col_count, &l_num_elements) == 3 ) {
          l_header_read = 1;
          desc.row_count = l_row_count;
          desc.col_count = l_col_count;
          desc.num_elements = l_num_elements;
        } else {
          fprintf( stderr, "could not csr description!\n" );
          return desc;
        }
      } else {
      }
    }
  }

  return desc;
}

static edge_mat_desc libxsmm_sparse_csc_reader_desc( const char*    i_csc_file_in ) {
  FILE *l_csc_file_handle;
  const unsigned int l_line_length = 512;
  char l_line[512/*l_line_length*/+1];
  unsigned int l_header_read = 0;
  unsigned int l_row_count = 0;
  unsigned int l_col_count = 0;
  unsigned int l_num_elements = 0;
  edge_mat_desc desc;

  desc.row_count = 0;
  desc.col_count = 0;
  desc.num_elements = 0;

  l_csc_file_handle = fopen( i_csc_file_in, "r" );
  if ( l_csc_file_handle == NULL ) {
    fprintf( stderr, "cannot open CSC file!\n" );
    return desc;
  }

  while (fgets(l_line, l_line_length, l_csc_file_handle) != NULL) {
    if ( strlen(l_line) == l_line_length ) {
      fprintf( stderr, "could not read file length!\n" );
      return desc;
    }
    /* check if we are still reading comments header */
    if ( l_line[0] == '%' ) {
      continue;
    } else {
      /* if we are the first line after comment header, we allocate our data structures */
      if ( l_header_read == 0 ) {
        if ( sscanf(l_line, "%u %u %u", &l_row_count, &l_col_count, &l_num_elements) == 3 ) {
          l_header_read = 1;
          desc.row_count = l_row_count;
          desc.col_count = l_col_count;
          desc.num_elements = l_num_elements;
        } else {
          fprintf( stderr, "could not csc description!\n" );
          return desc;
        }
      } else {
      }
    }
  }

  return desc;
}

