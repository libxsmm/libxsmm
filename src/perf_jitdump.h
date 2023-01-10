/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Maciej Debski (Google Inc.)
******************************************************************************/
#ifndef PERF_JITDUMP_H
#define PERF_JITDUMP_H

#if defined(PERF_JITDUMP_NOLIBXSMM)
# define LIBXSMM_EXTERN_C
# define PERF_JITDUMP_GLOBAL_VARIABLE(VARIABLE, INIT) VARIABLE = (INIT)
#else
# include <libxsmm_macros.h>
# define PERF_JITDUMP_GLOBAL_VARIABLE(VARIABLE, INIT) LIBXSMM_APIVAR_PRIVATE(VARIABLE)
#endif


LIBXSMM_EXTERN_C typedef struct jitdump_file_header {
  uint32_t magic;
  uint32_t version;
  uint32_t total_size;
  uint32_t elf_mach;
  uint32_t pad1;
  uint32_t pid;
  uint64_t timestamp;
  uint64_t flags;
} jitdump_file_header;


LIBXSMM_EXTERN_C typedef struct jitdump_record_header {
  uint32_t id;
  uint32_t total_size;
  uint64_t timestamp;
} jitdump_record_header;


LIBXSMM_EXTERN_C typedef struct jitdump_record_code_load {
  uint32_t pid;
  uint32_t tid;
  uint64_t vma;
  uint64_t code_addr;
  uint64_t code_size;
  uint64_t code_index;
  /* Needs to be followed with 0-terminated function name and raw native code */
} jitdump_record_code_load;


LIBXSMM_EXTERN_C typedef struct jitdump_record_code_move {
  uint32_t pid;
  uint32_t tid;
  uint64_t vma;
  uint64_t old_code_addr;
  uint64_t new_code_addr;
  uint64_t code_size;
  uint64_t code_index;
} jitdump_record_code_move;


LIBXSMM_EXTERN_C typedef struct jitdump_debug_entry {
  uint64_t code_addr;
  uint32_t line;
  uint32_t discrim;
  /* Followed by 0-terminated source file name. */
} jitdump_debug_entry;


LIBXSMM_EXTERN_C typedef struct jitdump_record_code_debug_info {
  uint64_t code_addr;
  uint64_t nr_entry;
  /* Followed by nr_entry jitdump_debug_entry structures. */
} jitdump_record_code_debug_info;


/* Currently empty */
LIBXSMM_EXTERN_C typedef struct jitdump_record_code_close {
  int dummy; /* avoid warning about struct without member */
} jitdump_record_code_close;


/* magic is "JiTD", serialized differently dependent on endianness. */
PERF_JITDUMP_GLOBAL_VARIABLE(/*const*/ uint32_t JITDUMP_MAGIC, 'J' << 24 | 'i' << 16 | 'T' << 8 | 'D');
PERF_JITDUMP_GLOBAL_VARIABLE(/*const*/ uint32_t JITDUMP_MAGIC_SWAPPED, 'J' | 'i' << 8 | 'T' << 16 | 'D' << 24);
PERF_JITDUMP_GLOBAL_VARIABLE(/*const*/ uint32_t JITDUMP_VERSION, 1);
PERF_JITDUMP_GLOBAL_VARIABLE(/*const*/ uint64_t JITDUMP_FLAGS_ARCH_TIMESTAMP, 1ULL);
PERF_JITDUMP_GLOBAL_VARIABLE(/*const*/ uint32_t JITDUMP_CODE_LOAD, 0);
PERF_JITDUMP_GLOBAL_VARIABLE(/*const*/ uint32_t JITDUMP_CODE_MOVE, 1);
PERF_JITDUMP_GLOBAL_VARIABLE(/*const*/ uint32_t JITDUMP_CODE_DEBUG_INFO, 2);
PERF_JITDUMP_GLOBAL_VARIABLE(/*const*/ uint32_t JITDUMP_CODE_CLOSE, 3);

#endif /* PERF_JITDUMP_H */
