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
#include "libxsmm_perf.h"
#include <libxsmm_sync.h>

#include "perf_jitdump.h"
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
#include "libxsmm_memory.h"
# include <sys/mman.h>
# include <sys/types.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <fcntl.h>
# include <errno.h>
# include <time.h>
#endif
#if defined(__linux__)
# include <syscall.h>
#endif
#if defined(_WIN32)
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
#endif

#if !defined(NDEBUG)
# define LIBXSMM_PERF_ERROR(msg) fprintf(stderr, msg)
#else
# define LIBXSMM_PERF_ERROR(msg)
#endif

#if !defined(PERF_JITDUMP_NOLIBXSMM)
LIBXSMM_APIVAR_PRIVATE_DEF(/*const*/ uint32_t JITDUMP_MAGIC);
LIBXSMM_APIVAR_PRIVATE_DEF(/*const*/ uint32_t JITDUMP_MAGIC_SWAPPED);
LIBXSMM_APIVAR_PRIVATE_DEF(/*const*/ uint32_t JITDUMP_VERSION);
LIBXSMM_APIVAR_PRIVATE_DEF(/*const*/ uint64_t JITDUMP_FLAGS_ARCH_TIMESTAMP);
LIBXSMM_APIVAR_PRIVATE_DEF(/*const*/ uint32_t JITDUMP_CODE_LOAD);
LIBXSMM_APIVAR_PRIVATE_DEF(/*const*/ uint32_t JITDUMP_CODE_MOVE);
LIBXSMM_APIVAR_PRIVATE_DEF(/*const*/ uint32_t JITDUMP_CODE_DEBUG_INFO);
LIBXSMM_APIVAR_PRIVATE_DEF(/*const*/ uint32_t JITDUMP_CODE_CLOSE);
#endif

LIBXSMM_APIVAR_DEFINE(FILE* internal_perf_fp);
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
LIBXSMM_APIVAR_DEFINE(libxsmm_timer_tickint (*internal_perf_timer)(void));
LIBXSMM_APIVAR_DEFINE(void* internal_perf_marker);
LIBXSMM_APIVAR_DEFINE(int internal_perf_codeidx);
#endif


LIBXSMM_API_INTERN void libxsmm_perf_init(libxsmm_timer_tickint (*timer_tick)(void))
{
  const uint32_t pid = (uint32_t)libxsmm_get_pid();
  char file_name[LIBXSMM_MAX_PATH] = "";
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
  char file_path[LIBXSMM_MAX_PATH];
  int fd, page_size, res;
  struct jitdump_file_header header;
  char * path_base;
  char date[64];
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);

  LIBXSMM_ASSERT(NULL != timer_tick);
  internal_perf_timer = timer_tick;

  /* initialize global variables */
  JITDUMP_MAGIC = ('J' << 24 | 'i' << 16 | 'T' << 8 | 'D');
  JITDUMP_MAGIC_SWAPPED = ('J' | 'i' << 8 | 'T' << 16 | 'D' << 24);
  JITDUMP_VERSION = 1;
  JITDUMP_FLAGS_ARCH_TIMESTAMP = 1ULL /* << 0 */;
  JITDUMP_CODE_LOAD = 0;
  JITDUMP_CODE_MOVE = 1;
  JITDUMP_CODE_DEBUG_INFO = 2;
  JITDUMP_CODE_CLOSE = 3;

  path_base = getenv("JITDUMPDIR");
  if (path_base == NULL) {
    path_base = getenv("HOME");
  }
  if (path_base == NULL) {
    path_base = ".";
  }

  LIBXSMM_SNPRINTF(file_path, sizeof(file_path), "%s/.debug/", path_base);
  res = mkdir(file_path, S_IRWXU);
  if (res < 0 && errno != EEXIST) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to create .debug dir\n");
    goto error;
  }

  LIBXSMM_SNPRINTF(file_path, sizeof(file_path), "%s/.debug/jit", path_base);
  res = mkdir(file_path, S_IRWXU);
  if (res < 0 && errno != EEXIST) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to create .debug/jit dir\n");
    goto error;
  }

  strftime(date, sizeof(date), "%Y%m%d", &tm);

  LIBXSMM_SNPRINTF(file_path, sizeof(file_path),
           "%s/.debug/jit/libxsmm-jit-%s.XXXXXX", path_base, date);
  path_base = mkdtemp(file_path);
  if (path_base == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to create temporary dir\n");
    goto error;
  }

  LIBXSMM_SNPRINTF(file_name, sizeof(file_name), "%s/jit-%u.dump", path_base, pid);

  fd = open(file_name, O_CREAT|O_TRUNC|O_RDWR, 0600);
  if (fd < 0) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to open file\n");
    goto error;
  }

  page_size = sysconf(_SC_PAGESIZE);
  if (page_size < 0) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to get page size\n");
    goto error;
  }
  internal_perf_marker = mmap(NULL, page_size, PROT_READ|PROT_EXEC, MAP_PRIVATE, fd, 0);
  if (internal_perf_marker == MAP_FAILED) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: mmap failed.\n");
    goto error;
  }

  /* initialize code index */
  internal_perf_codeidx = 0;

  internal_perf_fp = fdopen(fd, "wb+");
  if (internal_perf_fp == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: fdopen failed.\n");
    goto error;
  }

  LIBXSMM_MEMZERO127(&header);
  header.magic      = JITDUMP_MAGIC;
  header.version    = JITDUMP_VERSION;
  header.elf_mach   = 62;  /* EM_X86_64 */
  header.total_size = sizeof(header);
  header.pid        = pid;
  header.timestamp  = internal_perf_timer();
  header.flags      = JITDUMP_FLAGS_ARCH_TIMESTAMP;

  res = fwrite(&header, sizeof(header), 1, internal_perf_fp);
  if (res != 1) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to write header.\n");
    goto error;
  }
#else
  LIBXSMM_UNUSED(timer_tick);
  LIBXSMM_SNPRINTF(file_name, sizeof(file_name), "/tmp/perf-%u.map", pid);
  internal_perf_fp = fopen(file_name, "w+");
  if (internal_perf_fp == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to open map file\n");
    goto error;
  }
#endif
  return;
error:
  if (internal_perf_fp != NULL) {
    fclose(internal_perf_fp);
    internal_perf_fp = NULL;
  }
  assert(0);
}


LIBXSMM_API_INTERN void libxsmm_perf_finalize(void)
{
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
  int res, page_size;
  struct jitdump_record_header hdr;

  if (internal_perf_fp == NULL) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: jit dump file not opened\n");
    goto error;
  }

  LIBXSMM_MEMZERO127(&hdr);
  hdr.id = JITDUMP_CODE_CLOSE;
  hdr.total_size = sizeof(hdr);
  LIBXSMM_ASSERT(NULL != internal_perf_timer);
  hdr.timestamp = internal_perf_timer();
  res = fwrite(&hdr, sizeof(hdr), 1, internal_perf_fp);

  if (res != 1) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to write JIT_CODE_CLOSE record\n");
    goto error;
  }

  page_size = sysconf(_SC_PAGESIZE);
  if (page_size < 0) {
    LIBXSMM_PERF_ERROR("LIBXSMM ERROR: failed to get page_size\n");
    goto error;
  }
  munmap(internal_perf_marker, page_size);
  fclose(internal_perf_fp);
  return;
error:
  assert(0);
#else
  fclose(internal_perf_fp);
#endif
}


#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
/** Utility function to receive the OS-specific thread ID. */
LIBXSMM_API_INLINE unsigned int internal_perf_get_tid(void)
{
#if defined(__linux__)
  return (unsigned int)syscall(__NR_gettid);
#else /* fallback */
  return libxsmm_get_tid();
#endif
}
#endif


LIBXSMM_API_INTERN void libxsmm_perf_dump_code(const void* memory, size_t size, const char* name)
{
  LIBXSMM_ASSERT(internal_perf_fp != NULL);
  LIBXSMM_ASSERT(name != NULL && '\0' != *name);
  LIBXSMM_ASSERT(memory != NULL && size != 0);
  if (internal_perf_fp != NULL) {
#if defined(LIBXSMM_PERF_JITDUMP) && !defined(_WIN32)
    int res;
    struct jitdump_record_header hdr;
    struct jitdump_record_code_load rec;
    size_t name_len = strlen(name) + 1;

    LIBXSMM_MEMZERO127(&hdr);
    LIBXSMM_MEMZERO127(&rec);

    hdr.id = JITDUMP_CODE_LOAD;
    hdr.total_size = sizeof(hdr) + sizeof(rec) + name_len + size;
    LIBXSMM_ASSERT(NULL != internal_perf_timer);
    hdr.timestamp = internal_perf_timer();

    rec.code_size = size;
    rec.vma = (uintptr_t) memory;
    rec.code_addr = (uintptr_t) memory;
    rec.pid = (uint32_t) libxsmm_get_pid();
    rec.tid = (uint32_t) internal_perf_get_tid();

    LIBXSMM_FLOCK(internal_perf_fp);

    /* This will be unique as we hold the file lock. */
    rec.code_index = internal_perf_codeidx++;

    /* Count number of written items to check for errors. */
    res = 0;
    res += fwrite_unlocked(&hdr, sizeof(hdr), 1, internal_perf_fp);
    res += fwrite_unlocked(&rec, sizeof(rec), 1, internal_perf_fp);
    res += fwrite_unlocked(name, name_len, 1, internal_perf_fp);
    res += fwrite_unlocked((const void*) memory, size, 1, internal_perf_fp);

    LIBXSMM_FUNLOCK(internal_perf_fp);
    fflush(internal_perf_fp);

    LIBXSMM_ASSERT(res == 4); /* Expected 4 items written above */
#else
    fprintf(internal_perf_fp, "%" PRIxPTR " %lx %s\n", (uintptr_t)memory, (unsigned long)size, name);
    fflush(internal_perf_fp);
#endif
  }
}
