#include <libxsmm_source.h>
#include <vector>

int main(/*int argc, char* argv[]*/)
{
  typedef double value_type;
  int batchsize = 1000, m = 13, n = 5, k = 7;
  std::vector<value_type> a(batchsize*m*k), b(batchsize*k*n), c(m*n, 0);
  /* C/C++ and Fortran interfaces are available */
  typedef libxsmm_mmfunction<value_type> kernel_type;
  /* generates and dispatches a matrix multiplication kernel (C++ functor) */
  kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, m,n,k, 1.0/*alpha*/, 1.0/*beta*/);
  assert(kernel);
  for (int i = 0; i < batchsize; ++i) { /* initialize input */
    a[i*m*k] = static_cast<value_type>(1) / (i % 25);
    b[i*k*n] = static_cast<value_type>(7) / (i % 75);
  }
  /* kernel multiplies and accumulates matrix products: C += Ai * Bi */
  for (int i = 0; i < batchsize; ++i) kernel(&a[i*m*k], &b[i*k*n], &c[0]);
}
