/******************************************************************************
** Copyright (c) 2015-2016, Google Inc.
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

#include <stddef.h>
#include <stdint.h>

struct jitdump_file_header {
  uint32_t magic;
  uint32_t version;
  uint32_t total_size;
  uint32_t elf_mach;
  uint32_t pad1;
  uint32_t pid;
  uint64_t timestamp;
  uint64_t flags;
};

/* magic is "JiTD", serialized differently dependent on endianness. */
const uint32_t JITDUMP_MAGIC = 'J' << 24 | 'i' << 16 | 'T' << 8 | 'D';
const uint32_t JITDUMP_MAGIC_SWAPPED = 'J' | 'i' << 8 | 'T' << 16 | 'D' << 24;
const uint32_t JITDUMP_VERSION = 1;
const uint64_t JITDUMP_FLAGS_ARCH_TIMESTAMP = 1ULL<<0;

struct jitdump_record_header {
  uint32_t id;
  uint32_t total_size;
  uint64_t timestamp;
};

const uint32_t JITDUMP_CODE_LOAD = 0;
const uint32_t JITDUMP_CODE_MOVE = 1;
const uint32_t JITDUMP_CODE_DEBUG_INFO = 2;
const uint32_t JITDUMP_CODE_CLOSE = 3;

struct jitdump_record_code_load {
  uint32_t pid;
  uint32_t tid;
  uint64_t vma;
  uint64_t code_addr;
  uint64_t code_size;
  uint64_t code_index;
  /* Needs to be followed with 0-terminated function name and raw native code */
};

struct jitdump_record_code_move {
  uint32_t pid;
  uint32_t tid;
  uint64_t vma;
  uint64_t old_code_addr;
  uint64_t new_code_addr;
  uint64_t code_size;
  uint64_t code_index;
};

struct jitdump_debug_entry {
  uint64_t code_addr;
  uint32_t line;
  uint32_t discrim;
  /* Followed by 0-terminated source file name. */
};

struct jitdump_record_code_debug_info {
  uint64_t code_addr;
  uint64_t nr_entry;
  /* Followed by nr_entry jitdump_debug_entry structures. */
};

/* Currently empty */
struct jitdump_record_code_close {
};

#endif /* PERF_JITDUMP_H */
