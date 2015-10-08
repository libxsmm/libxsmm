#include "libxsmm_dispatch.h"
#include "libxsmm_crc32.h"
#include <libxsmm.h>

#define LIBXSMM_CACHESIZE (LIBXSMM_MAX_M) * (LIBXSMM_MAX_N) * (LIBXSMM_MAX_K) * 24
#define LIBXSMM_SEED 0


/** Filled with zeros due to C language rule. */
LIBXSMM_RETARGETABLE const void* libxsmm_cache[2][(LIBXSMM_CACHESIZE)];
LIBXSMM_RETARGETABLE int libxsmm_init = 0;


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE const void* libxsmm_dispatch(const void* key, size_t key_size, size_t cache_id, const void* value)
{
  const unsigned int hash = libxsmm_crc32(key, key_size, LIBXSMM_SEED), i = hash % (LIBXSMM_CACHESIZE);
  const void* *const cache = libxsmm_cache[cache_id%2];
  const void *const prev_value = cache[i];
  cache[i] = value;
  return prev_value;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE const void* libxsmm_lookup(const void* key, size_t key_size, size_t cache_id)
{
  return libxsmm_cache[cache_id%2][libxsmm_crc32(key, key_size, LIBXSMM_SEED)%(LIBXSMM_CACHESIZE)];
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smm_function libxsmm_smm_dispatch(int m, int n, int k)
{
  struct { int m, n, k; } args = { m, n, k };

  if (0 == libxsmm_init) {
    libxsmm_initialize();
    libxsmm_init = 1;
  }

  return libxsmm_lookup(&args, sizeof(args), 0);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmm_function libxsmm_dmm_dispatch(int m, int n, int k)
{
  struct { int m, n, k; } args = { m, n, k };

  if (0 == libxsmm_init) {
    libxsmm_initialize();
    libxsmm_init = 1;
  }

  return libxsmm_lookup(&args, sizeof(args), 1);
}
