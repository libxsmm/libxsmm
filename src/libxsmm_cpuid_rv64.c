/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm_cpuid.h>
#include <libxsmm_generator.h>
#include <libxsmm_sync.h>
#include "libxsmm_main.h"
#include "riscv_vector.h"

#include <signal.h>
#include <setjmp.h>

LIBXSMM_API int libxsmm_cpuid_rv64(libxsmm_cpuid_info* LIBXSMM_ARGDEF(info, NULL))
{
  int mvl;

  switch (__riscv_vsetvl_e8m1 (65536) * 8){
    case 128:
      mvl = LIBXSMM_RV64_MVL128;
      break;

    case 256:
      mvl = LIBXSMM_RV64_MVL256;
      break;
    default:
      mvl = LIBXSMM_RV64_MVL128;
      break;
  }

  return mvl;
}
