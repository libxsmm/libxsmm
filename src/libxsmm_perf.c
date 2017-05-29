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
#include "libxsmm_perf.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include "perf_jitdump.h"
#include <stdio.h>
#include <stdlib.h>
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
# include <sys/mman.h>
# include <string.h>
# include <sys/types.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <fcntl.h>
# include <errno.h>
# include <time.h>
#endif
#if defined(_WIN32)
# include <windows.h>
# define LIBXSMM_MAX_PATH MAX_PATH
#else
# if defined(__linux__)
#   include <linux/limits.h>
#   define LIBXSMM_MAX_PATH PATH_MAX
# elif defined(PATH_MAX)
#   define LIBXSMM_MAX_PATH PATH_MAX
# else /* fallback */
#   define LIBXSMM_MAX_PATH 1024
# endif
# include <unistd.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(NDEBUG)
# define LIBXSMM_PERF_ERROR(msg) fprintf(stderr, msg)
#else
# define LIBXSMM_PERF_ERROR(msg)
#endif


LIBXSMM_API_INTERN FILE * fp;
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
LIBXSMM_API_INTERN void* marker_addr;
LIBXSMM_API_INTERN int code_index /*= 0*/;
#endif


LIBXSMM_API_DEFINITION void libxsmm_perf_init(void)
{
  const uint32_t pid = (uint32_t)libxsmm_get_pid();
  char file_name[LIBXSMM_MAX_PATH];
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
  char file_path[LIBXSMM_MAX_PATH];
  int fd, page_size, res;
  struct jitdump_file_header header;
  char * path_base;
  char date[64];
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);

  /* initialize global variables */
  JITDUMP_MAGIC = 'J' << 24 | 'i' << 16 | 'T' << 8 | 'D';
  JITDUMP_MAGIC_SWAPPED = 'J' | 'i' << 8 | 'T' << 16 | 'D' << 24;
  JITDUMP_VERSION = 1;
  JITDUMP_FLAGS_ARCH_TIMESTAMP = 1ULL /*<< 0*/;
  JITDUMP_CODE_LOAD = 0;
  JITDUMP_CODE_MOVE = 1;
  JITDUMP_CODE_DEBUG_INFO = 2;
  JITDUMP_CODE_CLOSE = 3;

  path_base = getenv("JITDUMPDIR");
  if (path_base == NULL) {
    path_base = getenv("HOME");
  }
  if (path_base == NULL) {
    path_base = getenv(".");
  }

  LIBXSMM_SNPRINTF(file_path, sizeof(file_path), "%s/.debug/", path_base);
  res = mkdir(file_path, S_IRWXU);
  if (res < 0 && errno != EEXIST) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to create .debug dir\n");
    goto error;
  }

  LIBXSMM_SNPRINTF(file_path, sizeof(file_path), "%s/.debug/jit", path_base);
  res = mkdir(file_path, S_IRWXU);
  if (res < 0 && errno != EEXIST) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to create .debug/jit dir\n");
    goto error;
  }

  strftime(date, sizeof(date), "%Y%m%d", &tm);

  LIBXSMM_SNPRINTF(file_path, sizeof(file_path),
           "%s/.debug/jit/libxsmm-jit-%s.XXXXXX", path_base, date);
  path_base = mkdtemp(file_path);
  if (path_base == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to create temporary dir\n");
    goto error;
  }

  LIBXSMM_SNPRINTF(file_name, sizeof(file_name), "%s/jit-%u.dump", path_base, pid);

  fd = open(file_name, O_CREAT|O_TRUNC|O_RDWR, 0600);
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

  memset(&header, 0, sizeof(header));
  header.magic      = JITDUMP_MAGIC;
  header.version    = JITDUMP_VERSION;
  header.elf_mach   = 62;  /* EM_X86_64 */
  header.total_size = sizeof(header);
  header.pid        = pid;
  header.timestamp  = libxsmm_timer_xtick();
  header.flags      = JITDUMP_FLAGS_ARCH_TIMESTAMP;

  res = fwrite(&header, sizeof(header), 1, fp);
  if (res != 1) {
    LIBXSMM_PERF_ERROR("LIBXSMM: failed to write header.\n");
    goto error;
  }

#else
  LIBXSMM_SNPRINTF(file_name, sizeof(file_name), "/tmp/perf-%u.map", pid);
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


LIBXSMM_API_DEFINITION void libxsmm_perf_finalize(void)
{
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
  int res, page_size;
  struct jitdump_record_header hdr;

  if (fp == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM: jit dump file not opened\n");
    goto error;
  }

  memset(&hdr, 0, sizeof(hdr));
  hdr.id = JITDUMP_CODE_CLOSE;
  hdr.total_size = sizeof(hdr);
  hdr.timestamp = libxsmm_timer_xtick();
  res = fwrite(&hdr, sizeof(hdr), 1, fp);

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


LIBXSMM_API_DEFINITION void libxsmm_perf_dump_code(const void* memory, size_t size, const char* name)
{
  assert(fp != NULL);
  assert(name && *name);
  assert(memory != NULL && size != 0);
  if (fp != NULL) {
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
    int res;
    struct jitdump_record_header hdr;
    struct jitdump_record_code_load rec;
    size_t name_len = strlen(name) + 1;

    memset(&hdr, 0, sizeof(hdr));
    memset(&rec, 0, sizeof(rec));

    hdr.id = JITDUMP_CODE_LOAD;
    hdr.total_size = sizeof(hdr) + sizeof(rec) + name_len + size;
    hdr.timestamp = libxsmm_timer_xtick();

    rec.code_size = size;
    rec.vma = (uintptr_t) memory;
    rec.code_addr = (uintptr_t) memory;
    rec.pid = (uint32_t) libxsmm_get_pid();
    rec.tid = (uint32_t) libxsmm_get_tid_os();

#if !defined(LIBXSMM_NO_SYNC)
    flockfile(fp);
#endif

    /* This will be unique as we hold the file lock. */
    rec.code_index = code_index++;

    /* Count number of written items to check for errors. */
    res = 0;
    res += fwrite_unlocked(&hdr, sizeof(hdr), 1, fp);
    res += fwrite_unlocked(&rec, sizeof(rec), 1, fp);
    res += fwrite_unlocked(name, name_len, 1, fp);
    res += fwrite_unlocked((const void*) memory, size, 1, fp);

#if !defined(LIBXSMM_NO_SYNC)
    funlockfile(fp);
#endif

    fflush(fp);

    assert(res == 4); /* Expected 4 items written above */

#else
    fprintf(fp, "%llx %lx %s\n", (unsigned long long)((uintptr_t)memory), (unsigned long)size, name);
    fflush(fp);
#endif
  }
}

