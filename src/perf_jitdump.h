/******************************************************************************
** Copyright (c) 2015-2017, Google Inc.                                      **
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
/* Maciej Debski (Google Inc.)
******************************************************************************/
#ifndef PERF_JITDUMP_H
#define PERF_JITDUMP_H

#if defined(PERF_JITDUMP_NOLIBXSMM)
# define LIBXSMM_RETARGETABLE
# define LIBXSMM_API_INTERN
# define PERF_JITDUMP_GLOBAL_VARIABLE(VARIABLE, INIT) VARIABLE = (VALUE)
#else
# include <libxsmm_macros.h>
# define PERF_JITDUMP_GLOBAL_VARIABLE(VARIABLE, INIT) VARIABLE
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stddef.h>
#include <stdint.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


typedef struct LIBXSMM_RETARGETABLE jitdump_file_header {
  uint32_t magic;
  uint32_t version;
  uint32_t total_size;
  uint32_t elf_mach;
  uint32_t pad1;
  uint32_t pid;
  uint64_t timestamp;
  uint64_t flags;
} jitdump_file_header;


typedef struct LIBXSMM_RETARGETABLE jitdump_record_header {
  uint32_t id;
  uint32_t total_size;
  uint64_t timestamp;
} jitdump_record_header;


typedef struct LIBXSMM_RETARGETABLE jitdump_record_code_load {
  uint32_t pid;
  uint32_t tid;
  uint64_t vma;
  uint64_t code_addr;
  uint64_t code_size;
  uint64_t code_index;
  /* Needs to be followed with 0-terminated function name and raw native code */
} jitdump_record_code_load;


typedef struct LIBXSMM_RETARGETABLE jitdump_record_code_move {
  uint32_t pid;
  uint32_t tid;
  uint64_t vma;
  uint64_t old_code_addr;
  uint64_t new_code_addr;
  uint64_t code_size;
  uint64_t code_index;
} jitdump_record_code_move;


typedef struct LIBXSMM_RETARGETABLE jitdump_debug_entry {
  uint64_t code_addr;
  uint32_t line;
  uint32_t discrim;
  /* Followed by 0-terminated source file name. */
} jitdump_debug_entry;


typedef struct LIBXSMM_RETARGETABLE jitdump_record_code_debug_info {
  uint64_t code_addr;
  uint64_t nr_entry;
  /* Followed by nr_entry jitdump_debug_entry structures. */
} jitdump_record_code_debug_info;


/* Currently empty */
typedef struct LIBXSMM_RETARGETABLE jitdump_record_code_close {
  int dummy; /* avoid warning about struct without member */
} jitdump_record_code_close;


/* magic is "JiTD", serialized differently dependent on endianness. */
LIBXSMM_API_INTERN /*const*/ uint32_t PERF_JITDUMP_GLOBAL_VARIABLE(JITDUMP_MAGIC, 'J' << 24 | 'i' << 16 | 'T' << 8 | 'D');
LIBXSMM_API_INTERN /*const*/ uint32_t PERF_JITDUMP_GLOBAL_VARIABLE(JITDUMP_MAGIC_SWAPPED, 'J' | 'i' << 8 | 'T' << 16 | 'D' << 24);
LIBXSMM_API_INTERN /*const*/ uint32_t PERF_JITDUMP_GLOBAL_VARIABLE(JITDUMP_VERSION, 1);
LIBXSMM_API_INTERN /*const*/ uint64_t PERF_JITDUMP_GLOBAL_VARIABLE(JITDUMP_FLAGS_ARCH_TIMESTAMP, 1ULL /*<< 0*/);

LIBXSMM_API_INTERN /*const*/ uint32_t PERF_JITDUMP_GLOBAL_VARIABLE(JITDUMP_CODE_LOAD, 0);
LIBXSMM_API_INTERN /*const*/ uint32_t PERF_JITDUMP_GLOBAL_VARIABLE(JITDUMP_CODE_MOVE, 1);
LIBXSMM_API_INTERN /*const*/ uint32_t PERF_JITDUMP_GLOBAL_VARIABLE(JITDUMP_CODE_DEBUG_INFO, 2);
LIBXSMM_API_INTERN /*const*/ uint32_t PERF_JITDUMP_GLOBAL_VARIABLE(JITDUMP_CODE_CLOSE, 3);

#endif /* PERF_JITDUMP_H */
