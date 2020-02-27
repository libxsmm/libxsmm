#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <libxsmm.h>
#include "mdarray.hpp"
#include "rt_graph.hpp"

#if !defined(XSMM) && 1
# define XSMM
#endif
#if !defined(SCRATCH) && 1
# define SCRATCH
#endif
#if !defined(NAIVE2) && 0
# define NAIVE2
#endif


rt_graph::Timer timer;


template<typename T> void collocate_core(const int length_[3],
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
#if defined(SCRATCH)
    T *const Cdata = static_cast<T*>(libxsmm_aligned_scratch(sizeof(T) * co.size(0) * co.size(1) * length_[1], 0/*auto-alignment*/));
    T *const xyz_data = static_cast<T*>(libxsmm_aligned_scratch(sizeof(T) * co.size(0) * length_[0] * length_[1], 0/*auto-alignment*/));
    LIBXSMM_VLA_DECL(3, T, C, Cdata, co.size(1), length_[1]);
    LIBXSMM_VLA_DECL(3, T, xyz_alpha_beta, xyz_data, length_[0], length_[1]);
#else
    mdarray<T, 3, CblasRowMajor> C(co.size(0), co.size(1), length_[1]);
    mdarray<T, 3, CblasRowMajor> xyz_alpha_beta(co.size(0), length_[0], length_[1]);
#endif
#if defined(XSMM)
    const libxsmm_mmfunction<T> xmm1(LIBXSMM_GEMM_FLAG_NONE, length_[1], co.size(2), co.size(2),
      p_alpha_beta_reduced_.ld(), co.ld(), /*C.ld()*/length_[1],
      1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_AUTO);
    const libxsmm_mmfunction<T> xmm2(LIBXSMM_GEMM_FLAG_TRANS_B, length_[1], length_[0], co.size(2),
      /*C.ld()*/length_[1], p_alpha_beta_reduced_.ld(), /*xyz_alpha_beta.ld()*/length_[1],
      1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_AUTO);
    const libxsmm_mmfunction<T> xmm3(LIBXSMM_GEMM_FLAG_TRANS_B, length_[2], length_[0] * length_[1], co.size(2),
      p_alpha_beta_reduced_.ld(), /*xyz_alpha_beta.size(1)*/length_[0] * /*xyz_alpha_beta.ld()*/length_[1], ld,
      1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_NONE);
#endif
    timer.stop("init");
    timer.start("gemm");
    const T* bj = co.template at<CPU>(0, 0, 0);
#if defined(SCRATCH)
    T* cj = &LIBXSMM_VLA_ACCESS(3, C, 0, 0, 0, co.size(1), length_[1]);
#else
    T* cj = C.template at<CPU>(0, 0, 0);
#endif
    // run loop excluding the last element
    for (int a1 = 0; a1 < static_cast<int>(co.size(0) - 1); a1++) {
      const T *const bi = bj; bj = co.template at<CPU>(a1 + 1, 0, 0);
#if defined(SCRATCH)
      T *const ci = cj; cj = &LIBXSMM_VLA_ACCESS(3, C, a1 + 1, 0, 0, co.size(1), length_[1]);
#else
      T *const ci = cj; cj = C.template at<CPU>(a1 + 1, 0, 0);
#endif
#if defined(XSMM)
      xmm1(src_y, bi, ci, src_y, bj, cj);
#else
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        co.size(2), length_[1], co.size(2),
        1.0,
        bi, // Coef_{alpha,gamma,beta}
        co.ld(),
        src_y, // Y_{beta,j}
        p_alpha_beta_reduced_.ld(),
        0.0,
        ci, // tmp_{alpha, gamma, j}
        /*C.ld()*/length_[1]);
#endif
    }
    // execute remainder with pseudo-prefetch
#if defined(XSMM)
    xmm1(src_y, bj, cj, src_y, bj, cj);
#else
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      co.size(2), length_[1], co.size(2),
      1.0,
      bj, // Coef_{alpha,gamma,beta}
      co.ld(),
      src_y, // Y_{beta,j}
      p_alpha_beta_reduced_.ld(),
      0.0,
      cj, // tmp_{alpha, gamma, j}
      /*C.ld()*/length_[1]);
#endif

    // run loop excluding the last element
#if defined(SCRATCH)
    T* aj = &LIBXSMM_VLA_ACCESS(3, C, 0, 0, 0, co.size(1), length_[1]);
    cj = &LIBXSMM_VLA_ACCESS(3, xyz_alpha_beta, 0, 0, 0, length_[0], length_[1]);
#else
    T* aj = C.template at<CPU>(0, 0, 0);
    cj = xyz_alpha_beta.template at<CPU>(0, 0, 0);
#endif
    for (int a1 = 0; a1 < static_cast<int>(co.size(0) - 1); a1++) {
      T *const ai = aj, *const ci = cj;
#if defined(SCRATCH)
      aj = &LIBXSMM_VLA_ACCESS(3, C, a1 + 1, 0, 0, co.size(1), length_[1]);
      cj = &LIBXSMM_VLA_ACCESS(3, xyz_alpha_beta, a1 + 1, 0, 0, length_[0], length_[1]);
#else
      aj = C.template at<CPU>(a1 + 1, 0, 0);
      cj = xyz_alpha_beta.template at<CPU>(a1 + 1, 0, 0);
#endif
#if defined(XSMM)
      xmm2(ai, src_z, ci, aj, src_z, cj);
#else
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        length_[0], length_[1], co.size(2),
        1.0,
        src_z, // Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
        p_alpha_beta_reduced_.ld(),
        ai, // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
        /*C.ld()*/length_[1],
        0.0,
        ci, // contains xyz_{alpha, kj} the order kj is important
        /*xyz_alpha_beta.ld()*/length_[1]);
#endif
    }
    // execute remainder with pseudo-prefetch
#if defined(XSMM)
    xmm2(aj, src_z, cj, aj, src_z, cj);
#else
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      length_[0], length_[1], co.size(2),
      1.0,
      src_z, // Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
      p_alpha_beta_reduced_.ld(),
      aj, // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
      /*C.ld()*/length_[1],
      0.0,
      cj, // contains xyz_{alpha, kj} the order kj is important
      /*xyz_alpha_beta.ld()*/length_[1]);
#endif

#if defined(SCRATCH)
    cj = &LIBXSMM_VLA_ACCESS(3, xyz_alpha_beta, 0, 0, 0, length_[0], length_[1]);
#else
    cj = xyz_alpha_beta.template at<CPU>(0, 0, 0);
#endif
#if defined(XSMM)
    xmm3(src_x, cj, dst);
#else
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      length_[0] * length_[1], length_[2], co.size(2),
      1.0,
      cj,
      /*xyz_alpha_beta.size(1)*/length_[0] * /*xyz_alpha_beta.ld()*/length_[1],
      src_x,
      p_alpha_beta_reduced_.ld(),
      0.0,
      dst,
      ld);
#endif
    timer.stop("gemm");
#if defined(SCRATCH)
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
  mdarray<T, 3, CblasRowMajor> pol = mdarray<T, 3, CblasRowMajor>(3, lmax, std::max(std::max(i, j), k));
  mdarray<T, 3, CblasRowMajor> co = mdarray<T, 3, CblasRowMajor>(lmax, lmax, lmax);
  mdarray<T, 3, CblasRowMajor> Vgemm(i, j, k);
  mdarray<T, 3, CblasRowMajor> Vref(i, j, k);
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution(-1.0, 1.0);
  int length[3] = {i, j, k};
  for (int s = 0; s < static_cast<int>(pol.size()); s++)
    pol[s] = distribution(generator);

  for (int s = 0; s < static_cast<int>(co.size()); s++)
    co[s] = distribution(generator);

  Vgemm.zero();

  timer.start("collocate_gemm");
  collocate_core(length, co, pol, Vgemm);
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

