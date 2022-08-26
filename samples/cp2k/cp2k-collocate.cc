#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#if !defined(USE_HEADER_ONLY)
# include <libxsmm.h>
#else
# include <libxsmm_source.h>
#endif
#include "mdarray.hpp"
#include "rt_graph.hpp"

#if !defined(XSMM) && 1
# define XSMM
#endif
#if (defined(HAVE_MKL) || defined(__MKL) || defined(OPENBLAS) || defined(__OPENBLAS) || defined(__CBLAS)) && \
  !defined(TRIANGULAR) && 1
# define TRIANGULAR
#endif
#if !defined(SCRATCH) && 1
# define SCRATCH
#elif !defined(SCRATCH_LOCAL) && 1
# define SCRATCH_LOCAL
#endif
#if !defined(NAIVE2) && 0
# define NAIVE2
#endif


rt_graph::Timer timer;


template<typename T> void collocate_core(void* scratch, const int length_[3],
                     const mdarray<T, 3, CblasRowMajor> &co,
                     const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_reduced_,
                     mdarray<T, 3, CblasRowMajor> &Vtmp)
{
  const T *LIBXSMM_RESTRICT src_x = p_alpha_beta_reduced_.template at<CPU>(2, 0, 0);
  const T *LIBXSMM_RESTRICT src_y = p_alpha_beta_reduced_.template at<CPU>(1, 0, 0);
  const T *LIBXSMM_RESTRICT src_z = p_alpha_beta_reduced_.template at<CPU>(0, 0, 0);
  T *LIBXSMM_RESTRICT dst = Vtmp.template at<CPU>(0, 0, 0);
  const int ld = Vtmp.ld();

  if (co.size(0) > 1) {
    timer.start("init");
#if (defined(SCRATCH) || defined(SCRATCH_LOCAL))
# if defined(SCRATCH)
    T *const Cdata = LIBXSMM_ALIGN(static_cast<T*>(scratch), LIBXSMM_ALIGNMENT);
    T *const xyz_data = LIBXSMM_ALIGN(Cdata + co.size(0) * co.size(1) * length_[1], LIBXSMM_ALIGNMENT);
# else
    T *const Cdata = static_cast<T*>(libxsmm_aligned_scratch(sizeof(T) * co.size(0) * co.size(1) * length_[1], 0/*auto-alignment*/));
    T *const xyz_data = static_cast<T*>(libxsmm_aligned_scratch(sizeof(T) * co.size(0) * length_[0] * length_[1], 0/*auto-alignment*/));
# endif
# if defined(TRIANGULAR)
    mdarray<T, 2, CblasRowMajor> C(Cdata, co.size(1), length_[1]);
# else
    mdarray<T, 3, CblasRowMajor> C(Cdata, co.size(0), co.size(1), length_[1]);
# endif
    mdarray<T, 3, CblasRowMajor> xyz_alpha_beta(xyz_data, co.size(0), length_[0], length_[1]);
#else
# if defined(TRIANGULAR)
    mdarray<T, 2, CblasRowMajor> C(co.size(1), length_[1]);
# else
    mdarray<T, 3, CblasRowMajor> C(co.size(0), co.size(1), length_[1]);
# endif
    mdarray<T, 3, CblasRowMajor> xyz_alpha_beta(co.size(0), length_[0], length_[1]);
#endif
#if defined(XSMM)
    struct collocate {
      int i, j, k, lmax;
    } key = { static_cast<int>(Vtmp.size(0)), static_cast<int>(Vtmp.size(1)), static_cast<int>(Vtmp.size(2)), static_cast<int>(co.size(0)) };
    libxsmm_mmfunction<T>* kernelset = static_cast<libxsmm_mmfunction<T>*>(libxsmm_xdispatch(&key, sizeof(key)));
    if (NULL == kernelset) {
# if defined(TRIANGULAR)
      kernelset = static_cast<libxsmm_mmfunction<T>*>(libxsmm_xregister(&key, sizeof(key),
        sizeof(libxsmm_mmfunction<T>) * (static_cast<size_t>(2) * key.lmax - 1), NULL));
      for (int a1 = 0; a1 < (key.lmax - 1); a1++) {
        kernelset[2*a1+0] = libxsmm_mmfunction<T>(LIBXSMM_GEMM_FLAG_NONE,
          length_[1], static_cast<libxsmm_blasint>(co.size(1)) - a1, static_cast<libxsmm_blasint>(co.size(2)) - a1,
          static_cast<libxsmm_blasint>(p_alpha_beta_reduced_.ld()), static_cast<libxsmm_blasint>(co.ld()), static_cast<libxsmm_blasint>(C.ld()),
          1/*alpha*/, 0/*beta*/, LIBXSMM_GEMM_PREFETCH_AL2/*_AHEAD*/);
        kernelset[2*a1+1] = libxsmm_mmfunction<T>(LIBXSMM_GEMM_FLAG_TRANS_B, length_[1], length_[0], static_cast<libxsmm_blasint>(co.size(2)) - a1,
          static_cast<libxsmm_blasint>(C.ld()), static_cast<libxsmm_blasint>(p_alpha_beta_reduced_.ld()), static_cast<libxsmm_blasint>(xyz_alpha_beta.ld()),
          1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_NONE);
      }
      kernelset[2*(key.lmax-1)] = libxsmm_mmfunction<T>(LIBXSMM_GEMM_FLAG_TRANS_B,
        length_[2], length_[0] * length_[1], static_cast<libxsmm_blasint>(co.size(2)),
        static_cast<libxsmm_blasint>(p_alpha_beta_reduced_.ld()),
        static_cast<libxsmm_blasint>(xyz_alpha_beta.size(1)) * static_cast<libxsmm_blasint>(xyz_alpha_beta.ld()),
        ld, 1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_NONE);
# else
      kernelset = static_cast<libxsmm_mmfunction<T>*>(libxsmm_xregister(&key, sizeof(key), 3 * sizeof(libxsmm_mmfunction<T>), NULL));
      kernelset[0] = libxsmm_mmfunction<T>(LIBXSMM_GEMM_FLAG_NONE,
        length_[1], static_cast<libxsmm_blasint>(co.size(2)), static_cast<libxsmm_blasint>(co.size(2)),
        static_cast<libxsmm_blasint>(p_alpha_beta_reduced_.ld()), static_cast<libxsmm_blasint>(co.ld()), static_cast<libxsmm_blasint>(C.ld()),
        1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_AUTO);
      kernelset[1] = libxsmm_mmfunction<T>(LIBXSMM_GEMM_FLAG_TRANS_B, length_[1], length_[0], static_cast<libxsmm_blasint>(co.size(2)),
        static_cast<libxsmm_blasint>(C.ld()), static_cast<libxsmm_blasint>(p_alpha_beta_reduced_.ld()), static_cast<libxsmm_blasint>(xyz_alpha_beta.ld()),
        1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_AUTO);
      kernelset[2] = libxsmm_mmfunction<T>(LIBXSMM_GEMM_FLAG_TRANS_B,
        length_[2], length_[0] * length_[1], static_cast<libxsmm_blasint>(co.size(2)), static_cast<libxsmm_blasint>(p_alpha_beta_reduced_.ld()),
        static_cast<libxsmm_blasint>(xyz_alpha_beta.size(1)) * static_cast<libxsmm_blasint>(xyz_alpha_beta.ld()),
        ld, 1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_NONE);
# endif
    }
#endif
    timer.stop("init");
    timer.start("gemm");
    const T* aj = co.template at<CPU>(0, 0, 0);
#if defined(TRIANGULAR)
    T* cj = xyz_alpha_beta.template at<CPU>(0, 0, 0);
    T *const bi = C.template at<CPU>(0, 0);
#else
    T* cj = C.template at<CPU>(0, 0, 0);
#endif
    // run loop excluding the last element
    for (int a1 = 0; a1 < static_cast<int>(co.size(0) - 1); a1++) {
      const T *const ai = aj; aj = co.template at<CPU>(a1 + 1, 0, 0);
#if defined(TRIANGULAR)
      T *const ci = cj; cj = xyz_alpha_beta.template at<CPU>(a1 + 1, 0, 0);
#else
      T *const ci = cj; cj = C.template at<CPU>(a1 + 1, 0, 0);
#endif
#if defined(TRIANGULAR)
# if defined(XSMM)
      kernelset[2*a1+0](src_y, ai, bi, src_y, aj, bi);
      kernelset[2*a1+1](bi, src_z, ci/*, bi, src_z, cj*/);
# else
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        co.size(1) - a1, length_[1], co.size(2) - a1,
        1.0,
        ai, // Coef_{alpha,gamma,beta}
        co.ld(),
        src_y, // Y_{beta,j}
        p_alpha_beta_reduced_.ld(),
        0.0,
        bi, // tmp_{alpha, gamma, j}
        C.ld());
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        length_[0], length_[1], co.size(2) - a1,
        1.0,
        src_z, // Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
        p_alpha_beta_reduced_.ld(),
        bi, // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
        C.ld(),
        0.0,
        ci, // contains xyz_{alpha, kj} the order kj is important
        xyz_alpha_beta.ld());
# endif
#elif defined(XSMM)
      kernelset[0](src_y, ai, ci, src_y, aj, cj);
#else
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        co.size(2), length_[1], co.size(2),
        1.0,
        ai, // Coef_{alpha,gamma,beta}
        co.ld(),
        src_y, // Y_{beta,j}
        p_alpha_beta_reduced_.ld(),
        0.0,
        ci, // tmp_{alpha, gamma, j}
        C.ld());
#endif
    }
    // execute remainder
#if !defined(TRIANGULAR)
# if defined(XSMM)
    kernelset[0](src_y, aj, cj, src_y, aj, cj); // with pseudo-prefetch
# else
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      co.size(2), length_[1], co.size(2),
      1.0,
      aj, // Coef_{alpha,gamma,beta}
      co.ld(),
      src_y, // Y_{beta,j}
      p_alpha_beta_reduced_.ld(),
      0.0,
      cj, // tmp_{alpha, gamma, j}
      C.ld());
# endif
    // run loop excluding the last element
    const T* bj = C.template at<CPU>(0, 0, 0);
    cj = xyz_alpha_beta.template at<CPU>(0, 0, 0);
    for (int a1 = 0; a1 < static_cast<int>(co.size(0) - 1); a1++) {
      const T* const bi = bj;
      T *const ci = cj;
      bj = C.template at<CPU>(a1 + 1, 0, 0);
      cj = xyz_alpha_beta.template at<CPU>(a1 + 1, 0, 0);
# if defined(XSMM)
      kernelset[1](bi, src_z, ci, bj, src_z, cj);
# else
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        length_[0], length_[1], co.size(2),
        1.0,
        src_z, // Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
        p_alpha_beta_reduced_.ld(),
        bi, // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
        C.ld(),
        0.0,
        ci, // contains xyz_{alpha, kj} the order kj is important
        xyz_alpha_beta.ld());
# endif
    }
    // execute remainder
# if defined(XSMM)
    kernelset[1](bj, src_z, cj, bj, src_z, cj); // with pseudo-prefetch
# else
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      length_[0], length_[1], co.size(2),
      1.0,
      src_z, // Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
      p_alpha_beta_reduced_.ld(),
      bj, // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
      C.ld(),
      0.0,
      cj, // contains xyz_{alpha, kj} the order kj is important
      xyz_alpha_beta.ld());
# endif
#else
    memset(xyz_alpha_beta.template at<CPU>(co.size(0) - 1, 0, 0), 0,
      sizeof(T) * length_[0] * xyz_alpha_beta.ld());
    cblas_dger(CblasRowMajor,
      length_[0], length_[1],
      co(co.size(0) - 1, 0, 0),
      src_z, 1, src_y, 1,
      xyz_alpha_beta.template at<CPU>(co.size(0) - 1, 0, 0),
      xyz_alpha_beta.ld());
#endif
#if defined(XSMM)
# if defined(TRIANGULAR)
    kernelset[2*(key.lmax-1)](src_x, xyz_alpha_beta.template at<CPU>(0, 0, 0), dst);
# else
    kernelset[2](src_x, xyz_alpha_beta.template at<CPU>(0, 0, 0), dst);
# endif
#else
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      length_[0] * length_[1], length_[2], co.size(2),
      1.0,
      xyz_alpha_beta.template at<CPU>(0, 0, 0),
      xyz_alpha_beta.size(1) * xyz_alpha_beta.ld(),
      src_x,
      p_alpha_beta_reduced_.ld(),
      0.0,
      dst,
      ld);
#endif
    timer.stop("gemm");
#if defined(SCRATCH_LOCAL)
    timer.start("deinit");
    libxsmm_free(Cdata);
    libxsmm_free(xyz_data);
    timer.stop("deinit");
#endif
  } else {
    timer.start("remainder");
    for (int z1 = 0; z1 < length_[0]; z1++) {
      const T tz = co(0, 0, 0) * src_z[z1];
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for (int y1 = 0; y1 < length_[1]; y1++) {
        const T tmp = tz * src_y[y1];
        LIBXSMM_PRAGMA_SIMD
        for (int x1 = 0; x1 < length_[2]; x1++) {
          dst[x1] = tmp * src_x[x1];
        }
        dst += ld;
      }
    }
    timer.stop("remainder");
  }
}


template <typename T> void collocate_core_naive(const int *length_,
                        const mdarray<T, 3, CblasRowMajor> &co,
                        const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_reduced_,
                        mdarray<T, 3, CblasRowMajor> &Vtmp)
{
  Vtmp.zero();
  for (int alpha = 0; alpha < static_cast<int>(co.size(2)); alpha++) {
    for (int gamma = 0; gamma < static_cast<int>(co.size(0)); gamma++) {
      for (int beta = 0; beta < static_cast<int>(co.size(1)); beta++) {
        double coef = co(alpha, gamma, beta);
        for (int z = 0; z < length_[0]; z++) {
          double c1 = coef * p_alpha_beta_reduced_(0, gamma, z);
          for (int y = 0; y < length_[1]; y++) {
            double c2 = c1 * p_alpha_beta_reduced_(1, beta, y);
            for (int x = 0; x < length_[2]; x++) {
              Vtmp(z, y, x) += c2 * p_alpha_beta_reduced_(2, alpha, x);
            }
          }
        }
      }
    }
  }
}


#if defined(NAIVE2)
template <typename T> void collocate_core_naive2(const int *length_,
                        const mdarray<T, 3, CblasRowMajor> &co,
                        const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_reduced_,
                        mdarray<T, 3, CblasRowMajor> &Vtmp)
{
  Vtmp.zero();
  for (int gamma = 0; gamma < static_cast<int>(co.size(0)); gamma++) {
    for (int beta = 0; beta < static_cast<int>(co.size(1)); beta++) {
      for (int z = 0; z < length_[0]; z++) {
        double c1 = p_alpha_beta_reduced_(0, gamma, z);
        for (int y = 0; y < length_[1]; y++) {
          double c2 = c1 * p_alpha_beta_reduced_(1, beta, y);
          for (int x = 0; x < length_[2]; x++) {
            T tmp = 0;
            for (int alpha = 0; alpha < static_cast<int>(co.size(2)); alpha++) {
              double coef = co(alpha, gamma, beta);
              tmp += coef * p_alpha_beta_reduced_(2, alpha, x);
            }
            Vtmp(z, y, x) += c2 * tmp;
          }
        }
      }
    }
  }
}
#endif


// The three first numbers are the grid size, the last one can be anything
template <typename T> T test_collocate_core(const int i, const int j, const int k, const int lmax)
{
#if defined(SCRATCH)
  void* const scratch = malloc(sizeof(T) * (static_cast<size_t>(lmax) * lmax * j + static_cast<size_t>(lmax) * i * j) + 2 * LIBXSMM_ALIGNMENT);
#else
  void* const scratch = NULL;
#endif
  mdarray<T, 3, CblasRowMajor> pol = mdarray<T, 3, CblasRowMajor>(3, lmax, std::max(std::max(i, j), k));
  mdarray<T, 3, CblasRowMajor> co = mdarray<T, 3, CblasRowMajor>(lmax, lmax, lmax);
  mdarray<T, 3, CblasRowMajor> Vgemm(i, j, k);
  mdarray<T, 3, CblasRowMajor> Vref(i, j, k);
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution(-1.0, 1.0);
  int length[3] = {i, j, k};
  for (int s = 0; s < static_cast<int>(pol.size()); s++)
    pol[s] = distribution(generator);
#if !defined(TRIANGULAR)
  for (int s = 0; s < static_cast<int>(co.size()); s++)
    co[s] = distribution(generator);
#else
  co.zero();
  for (int a1 = 0; a1 < static_cast<int>(co.size(0)); a1++) {
    // for fixed a1, the matrix should be triangular of this form
    // b1 b2 b3
    // b4 b5
    // b6
    const int b2 = static_cast<int>(co.size(1)) - a1;
    for (int b1 = 0; b1 < b2; b1++) {
      for (int g1 = 0; g1 < (b2 - b1); g1++) {
        co(a1, b1, g1) = distribution(generator);
      }
    }
  }
#endif
  Vgemm.zero();

  timer.start("collocate_gemm");
  collocate_core(scratch, length, co, pol, Vgemm);
  timer.stop("collocate_gemm");

  timer.start("collocate_brute_force");
#if !defined(NAIVE2)
  collocate_core_naive(length,
             co,
             pol,
             Vref);
#else // variant
  collocate_core_naive2(length,
             co,
             pol,
             Vref);
#endif
  timer.stop("collocate_brute_force");

  T maxi = -2.0;
  for (int l = 0; l < static_cast<int>(Vgemm.size(0)); l++)
    for (int m = 0; m < static_cast<int>(Vgemm.size(1)); m++)
      for (int n = 0; n < static_cast<int>(Vgemm.size(2)); n++)
        maxi = std::max(std::abs(Vref(l, m, n) - Vgemm(l, m, n)), maxi);

  pol.clear();
  co.clear();
  Vgemm.clear();
  Vref.clear();
#if defined(SCRATCH)
  free(scratch);
#endif
  return maxi;
}


// template <typename T> void integrate_core_naive(const int *length_,
//                        const mdarray<T, 3, CblasRowMajor> &pol_,
//                        const mdarray<T, 3, CblasRowMajor> &Vtmp,
//                        mdarray<T, 3, CblasRowMajor> &co)
// {
//  for (int gamma = 0; gamma < co.size(0); gamma++) {
//    for (int beta = 0; beta < co.size(1); beta++) {
//      for (int alpha = 0; alpha < co.size(2); alpha++) {
//        T res = 0.0;
//        for (int z = 0; z < length_[0]; z++) {
//          for (int y = 0; y < length_[1]; y++) {
//            const T c1 = pol_(0, gamma, z) * pol_(1, beta, y);
//            const T*LIBXSMM_RESTRICT vtmp = Vtmp.template at<CPU>(z, y, 0);
//            for (int x = 0; x < length_[2]; x++) {
//              res += c1 * pol_(2, alpha, x) * vtmp[x];
//            }
//          }
//        }
//        co(gamma, beta, alpha) = res;
//      }
//    }
//  }
// }


// template <typename T> bool test_integrate_core(const int i, const int j, const int k, const int lmax)
// {
//  mdarray<T, 3, CblasRowMajor> pol = mdarray<T, 3, CblasRowMajor>(3,
//                                  lmax,
//                                  std::max(std::max(i, j), k));
//  mdarray<T, 3, CblasRowMajor> co_ref = mdarray<T, 3, CblasRowMajor>(lmax, lmax, lmax);
//  mdarray<T, 3, CblasRowMajor> co_gemm = mdarray<T, 3, CblasRowMajor>(lmax, lmax, lmax);
//  mdarray<T, 3, CblasRowMajor> V = mdarray<T, 3, CblasRowMajor>(i, j, k);
//  std::default_random_engine generator;
//  std::uniform_real_distribution<T> distribution(-1.0, 1.0);
//  int length[3] = {i, j, k};
//  for (int s = 0; s < pol.size(); s++)
//    pol[s] = distribution(generator);

//  for (int s = 0; s < V.size(); s++)
//    V[s] = distribution(generator);
//  co_gemm.zero();
//  integrate_core<T>(length, pol, V, co_gemm);
//  co_ref.zero();

//  integrate_core_naive<T>(length,
//              pol,
//              V,
//              co_ref);

//  T maxi = -2.0;
//  for (int l = 0; l < co_gemm.size(0); l++)
//    for (int m = 0; m < co_gemm.size(1); m++)
//      for (int n = 0; n < co_gemm.size(2); n++) {
//        maxi = std::max(std::abs(co_gemm(l, m, n) - co_ref(l, m, n)), maxi);
//      }

//  if (maxi > 1e-13)
//    return false;

//  return true;
// }


int main(int argc, char* argv[])
{
  typedef double elem_type;
  const int nrepin = (1 < argc ? atoi(argv[1]) : 100), nrep = std::max(nrepin, 1);
  const int n1in = (2 < argc ? atoi(argv[2]) : 0), n1 = std::max(n1in, 1);
  const int n2in = (3 < argc ? atoi(argv[3]) : n1), n2 = (0 < n2in ? n2in : n1);
  const int n3in = (4 < argc ? atoi(argv[4]) : n1), n3 = (0 < n3in ? n3in : n1);
  const int lmin = (5 < argc ? atoi(argv[5]) : 6), lmax = (0 < lmin ? lmin : 6);
  elem_type diff = 0;
#if (defined(HAVE_MKL) || defined(__MKL)) && 0
  mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
#endif
  timer.start("test_collocate_core");
  for (int i = 0; i < nrep; ++i) {
    if (0 == n1in) {
      diff = std::max(diff, test_collocate_core<elem_type>(27, 31, 23, 3));
      diff = std::max(diff, test_collocate_core<elem_type>(13, 35, 13, 7));
      diff = std::max(diff, test_collocate_core<elem_type>(15, 11, 23, 9));
      diff = std::max(diff, test_collocate_core<elem_type>(13, 19, 17, 5));
      diff = std::max(diff, test_collocate_core<elem_type>(9, 11, 19, 3));
      diff = std::max(diff, test_collocate_core<elem_type>(19, 17, 25, 5));
      diff = std::max(diff, test_collocate_core<elem_type>(23, 19, 27, 1));
      diff = std::max(diff, test_collocate_core<elem_type>(25, 23, 31, 11));
      diff = std::max(diff, test_collocate_core<elem_type>(27, 31, 23, 13));
    }
    else {
      diff = std::max(diff, test_collocate_core<elem_type>(n1, n2, n3, lmax));
    }
  }
  timer.stop("test_collocate_core");

  // process timings
  const auto result = timer.process();

  // print default statistics
  std::cout << "Default statistic:" << std::endl;
  std::cout << result.print();

  if (diff > 1e-14) {
    printf("Wrong result : maximum error %.15lf\n", diff);
    return 1;
  }
  return 0;
}

