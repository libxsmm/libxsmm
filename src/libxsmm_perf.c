/******************************************************************************
** Copyright (c) 2015-2016, Google Inc.                                      **
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
#include "libxsmm_perf.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(_WIN32)
# include <process.h>
# define LIBXSMM_PERF_GETPID _getpid
#else
# include <unistd.h>
# define LIBXSMM_PERF_GETPID getpid
#endif
#include <stdio.h>
#if defined(LIBXSMM_PERF_JITDUMP)
# include <string.h> /* included in front of <jitdump.h> */
/* make JITDUMP=/path/to/linux-kernel/tools/perf/util */
# include <jitdump.h>
# include <sys/mman.h>
# include <sys/types.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <syscall.h>
# include <fcntl.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(NDEBUG)
# define LIBXSMM_PERF_ERROR(msg) fprintf(stderr, msg)
#else
# define LIBXSMM_PERF_ERROR(msg)
#endif


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE FILE * fp;
#if defined(LIBXSMM_PERF_JITDUMP)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void* marker_addr;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int code_index /*= 0*/;

/* Records in jit dump file are padded to 8 bytes. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE char zeros[8];  /* for padding. */

LIBXSMM_INLINE LIBXSMM_RETARGETABLE size_t libxsmm_perf_padding_len(size_t len) {
  return PADDING_8ALIGNED(len);
}
#endif


LIBXSMM_API_DEFINITION void libxsmm_perf_init()
{
  const long pid = (long)LIBXSMM_PERF_GETPID();
  /* needs to hold "jit-<pid>.dump" or "perf-<pid>.map" */
  char file_name[64];
#if defined(LIBXSMM_PERF_JITDUMP)
  int fd, page_size, res;
  struct jitheader header;
  size_t padding_len;
  LIBXSMM_SNPRINTF(file_name, sizeof(file_name), "jit-%li.dump", pid);

  fd = open(file_name, O_CREAT|O_TRUNC|O_RDWR, 0666);
  if (fd < 0) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to open file\n");
    goto error;
  }

  page_size = sysconf(_SC_PAGESIZE);
  if (page_size < 0) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to get page size\n");
    goto error;
  }
  marker_addr = mmap(NULL, page_size, PROT_READ|PROT_EXEC, MAP_PRIVATE, fd, 0);
  if (marker_addr == MAP_FAILED) {
    LIBXSMM_PERF_ERROR("LIBXSMM: mmap failed.\n");
    goto error;
  }

  /* initialize code index */
  code_index = 0;

  fp = fdopen(fd, "wb+");
  if (fp == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM: fdopen failed.\n");
    goto error;
  }

  padding_len = libxsmm_perf_padding_len(sizeof(header));
  memset(&header, 0, sizeof(header));
  header.magic      = JITHEADER_MAGIC;
  header.version    = JITHEADER_VERSION;
  header.elf_mach   = 62;  /* EM_X86_64 */
  header.total_size = sizeof(header) + padding_len;
  header.pid        = pid;
  header.timestamp  = libxsmm_timer_xtick();
  header.flags      = JITDUMP_FLAGS_ARCH_TIMESTAMP;

  res = fwrite(&header, sizeof(header), 1, fp);
  if (res != 1) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to write header.\n");
    goto error;
  }
  if (padding_len > 0) {
    res = fwrite(&zeros, padding_len, 1, fp);
    if (res != 1) {
      LIBXSMM_PERF_ERROR("LIBXSMM: failed to write header padding.\n");
      goto error;
    }
  }

#else
  LIBXSMM_SNPRINTF(file_name, sizeof(file_name), "/tmp/perf-%li.map", pid);
  fp = fopen(file_name, "w+");
  if (fp == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to open map file\n");
    goto error;
  }
#endif

  return;

error:
  if (fp != NULL) {
    fclose(fp);
    fp = NULL;
  }
  assert(0);
}


LIBXSMM_API_DEFINITION void libxsmm_perf_finalize()
{
#if defined(LIBXSMM_PERF_JITDUMP)
  int res, page_size;
  struct jr_code_close rec;

  if (fp == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM: jit dump file not opened\n");
    goto error;
  }

  memset(&rec, 0, sizeof(rec));
  rec.p.id = JIT_CODE_CLOSE;
  rec.p.total_size = sizeof(rec);
  rec.p.timestamp = libxsmm_timer_xtick();
  res = fwrite(&rec, sizeof(rec), 1, fp);
  if (res != 1) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to write JIT_CODE_CLOSE record\n");
    goto error;
  }

  page_size = sysconf(_SC_PAGESIZE);
  if (page_size < 0) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to get page_size\n");
    goto error;
  }
  munmap(marker_addr, page_size);
  fclose(fp);
  return;

error:
  assert(0);
#else
  fclose(fp);
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_perf_write_code(const volatile void* memory, size_t size,
                             const char* name)
{
  assert(fp != NULL);
  assert(name && *name);
  assert(memory != NULL && size != 0);
  if (fp != NULL) {
#if defined(LIBXSMM_PERF_JITDUMP)
    int res, expected_res;
    struct jr_code_load rec;
    size_t name_len = strlen(name) + 1;
    size_t padding_len = libxsmm_perf_padding_len(sizeof(rec) + name_len);

    memset(&rec, 0, sizeof(rec));
    rec.p.id = JIT_CODE_LOAD;
    rec.p.total_size = sizeof(rec) + name_len + padding_len + size;
    rec.p.timestamp = libxsmm_timer_xtick();
    rec.code_size = size;
    rec.vma = (uintptr_t) memory;
    rec.code_addr = (uintptr_t) memory;
    rec.pid = LIBXSMM_PERF_GETPID();
    rec.tid = (pid_t) syscall(__NR_gettid);

#if !defined(LIBXSMM_NO_SYNC)
    flockfile(fp);
#endif

    rec.code_index = code_index++;

    /* Count number of written items to check for errors. */
    res = 0;
    expected_res = 3;
    res += fwrite_unlocked(&rec, sizeof(rec), 1, fp);
    res += fwrite_unlocked(name, name_len, 1, fp);
    if (padding_len > 0) {
      res += fwrite_unlocked(&zeros, padding_len, 1, fp);
      expected_res++;
    }
    res += fwrite_unlocked((const void*) memory, size, 1, fp);

#if !defined(LIBXSMM_NO_SYNC)
    funlockfile(fp);
#endif

    fflush(fp);

    assert(res == expected_res);

#else
    fprintf(fp, "%llx %lx %s\n", (unsigned long long)/*uintptr_t*/memory, (unsigned long)size, name);
    fflush(fp);
#endif
  }
}

