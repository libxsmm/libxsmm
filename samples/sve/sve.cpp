/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/

#include <libxsmm_source.h>
#include <vector>
#include <iostream>

// main directory:
// make BLAS=0 STATIC=0 -j 48


// g++ -DBLAS=0 -DLIBXSMM_NO_BLAS=1 -I../../include sve.cpp -L../../lib -lxsmm -pthread -o compiled
// LD_LIBRARY_PATH=../../lib LIBXSMM_VERBOSE=-1 ./compiled

// objdump -D -b binary -maarch64 libxsmm_x86_64_tsize1_20x5_20x5_opcode12_flags0_params29.meltw

int main(/*int argc, char* argv[]*/) {

  typedef float T;
  std::cout << "Hello world!" << std::endl;

  // somehow was only LIBXSMM_AARCH64_V82 even if it should have been the A64FX
  libxsmm_set_target_archid( LIBXSMM_AARCH64_A64FX );

  // goldig, endlich funktioniert es :)
  // also:
  // generate vector
  int size = 100;
  int size2 = size * 20;
  std::vector<T> a(size2), b(size2), c(size2);
  // fill vector with data
  for(int i=0;i<size;i++){
    a[i] = i;
    b[i] = 15 - (i & 15);
    c[i] = 1;
  }

  // generate functions, which we can call
  int m = (size / 5) & ~3, n = (size / m) & ~3;
  libxsmm_meltwfunction_unary set_zero = libxsmm_dispatch_meltw_unary(m, n, &m, &m, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  libxsmm_meltwfunction_unary copy     = libxsmm_dispatch_meltw_unary(m, n, &m, &m, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  libxsmm_meltwfunction_unary square   = libxsmm_dispatch_meltw_unary(m, n, &m, &m, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_X2);
  libxsmm_meltwfunction_unary trans    = libxsmm_dispatch_meltw_unary(m, n, &m, &n, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
  assert(set_zero);
  assert(copy);
  assert(square);
  assert(trans);

  if(set_zero == nullptr || copy == nullptr || square == nullptr || trans == nullptr){
    std::cerr << "generated kernel is null!!" << std::endl;
    return -1;
  }

  std::cout << "m: " << m << ", n: " << n << std::endl;

  std::cout << "created kernels" << std::endl;

  libxsmm_meltw_unary_param param;

  param.in.primary  = (void*) a.data();
  param.out.primary = (void*) b.data();

  set_zero( &param );
  std::cout << "set zero" << std::endl;
  
  param.in.primary  = (void*) a.data();
  param.out.primary = (void*) c.data();

  copy( &param );
  std::cout << "copy" << std::endl;
  
  param.in.primary  = (void*) a.data();
  param.out.primary = (void*) a.data();

  square( &param );
  std::cout << "square" << std::endl;

  // todo apply several functions on vector
  // todo compare vector with expected result
  for(int i=0;i<size;i++){
    std::cout << "i: " << i << ", a[i]: " << a[i] << ", b[i]: " << b[i] << ", c[i]: " << c[i] << std::endl;
  }
  std::cout << "done" << std::endl;
  // todo done (or free resources ^^)

}
