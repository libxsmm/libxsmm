/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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


int main(int argc, char* argv[])
{
  const char *const filename = (1 < argc ? argv[1] : "mhd_image.mhd");
  size_t ndims = 3, size[3], ncomponents, header_size, extension_size;
  libxsmm_mhd_elemtype type;
  char data_filename[1024];
  void* data = 0;
  int result;

  /* Read header information; function includes various sanity checks. */
  result = libxsmm_mhd_read_header(filename, sizeof(data_filename),
    data_filename, &ndims, size, &ncomponents, &type,
    &header_size, &extension_size);

  /* Allocate data according to the header information. */
  if (EXIT_SUCCESS == result) {
    size_t typesize;
    if (0 != libxsmm_mhd_typename(type, &typesize)) {
      const size_t nelements = size[0] * (1 < ndims ? (size[1] * (2 < ndims ? size[2] : 1)) : 1);
      data = malloc(ncomponents * typesize * nelements);
    }
    else {
      result = EXIT_FAILURE;
    }
  }

  /* Read the data according to the header into the allocated buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read(data_filename,
      size, size, ndims, ncomponents, header_size,
      type, 0/*type_data*/, data, 0/*handle_element*/,
      0/*extension*/, 0/*extension_size*/);
  }

  /* Write the data into a different file. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_write("mhd_test.mhd", size, size,
      ndims, ncomponents, type, data,
      0/*extension_header*/,
      0/*extension*/,
      0/*extension_size*/);
  }

  /* Read header information of newly written file. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read_header("mhd_test.mhd", sizeof(data_filename),
      data_filename, &ndims, size, &ncomponents, &type,
      &header_size, &extension_size);
  }

  /* Check the written data against the buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxsmm_mhd_read(data_filename,
      size, size, ndims, ncomponents, header_size,
      type, 0/*type_data*/, data, libxsmm_mhd_element_comparison,
      0/*extension*/, 0/*extension_size*/);
  }

  /* Deallocate the buffer. */
  free(data);

  return result;
}
