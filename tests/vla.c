#include <libxsmm_macros.h>
#include <stdlib.h>
#include <assert.h>


int main(/*int argc, char* argv[]*/)
{
  int ni = 9, nj = 7, nk = 3, i, j, k, linear = 0, result = EXIT_SUCCESS;
  int* in1 = (int*)malloc(ni * nj * nk * sizeof(int));
  LIBXSMM_VLA_DECL(3, const int, in3, in1, nj, nk);

  assert(0 != in1);
  for (i = 0; i < (ni * nj * nk); ++i) in1[i] = i;
  for (i = 0; i < ni; ++i) {
    for (j = 0; j < nj; ++j) {
      for (k = 0; k < nk; ++k) {
        if (in1[linear] != LIBXSMM_VLA_ACCESS(3, in3, i, j, k, nj, nk)) {
          result = EXIT_FAILURE;
          i = ni; j = nj;
          break;
        }
        ++linear;
      }
    }
  }

  free(in1);
  return result;
}
