/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/

#ifndef _TIMING_H_
#define _TIMING_H_

#define MAX_THREADS 200

static __inline__ unsigned long long rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

inline double getFreq()
{
  long long int s = rdtsc();
  sleep(1);
  long long int e = rdtsc();
  return (e - s) * 1.0;
}
extern double ifreq;

inline double getTime() {
  return rdtsc() * ifreq;
}

enum DebugTimer {
  BRGEMM,
  XPOSE,
  DROPOUT,
  LAYER_NORM,
  SOFTMAX,
  GELU,
  BIAS,
  VNNI,
  EW_COPY,
  EW_ADD,
  EW_SCL,
  EW_ZERO,
  EW_RED,
  OPTIM,
  LAST_TIMER
};

static const char *DebugTimerNames[] = {"BRGEMM", "XPOSE", "DROPOUT", "LYR_NRM", "SOFTMAX", "GELU", "BIAS", "VNNI", "COPY", "ADD", "SCALE", "ZERO", "REDUCE", "OPTIM", "LAST_TIMER"};
enum PassType {
  FWD,
  BWD
};

enum ScopeType {
  q_gemm,
  k_gemm,
  v_gemm,
  a_gemm,
  c_gemm,
  i_gemm,
  o_gemm,
  diq_gemm,
  dik_gemm,
  div_gemm,
  //dia_gemm,
  //dic_gemm,
  dica_gemm,
  dii_gemm,
  dio_gemm,
  dwqkv_gemm,
  //dwq_gemm,
  //dwk_gemm,
  //dwv_gemm,
  dwa_gemm,
  dwc_gemm,
  dwi_gemm,
  dwo_gemm,
  dqkv_bias,
  di_bias,
  do_bias,
  w_vnni,
  a_vnni,
  w_xpose,
  a_xpose,
  st_other,
  NUM_SCOPES
};

static const char *ScopeNames[] = {
  "q_gemm",
  "k_gemm",
  "v_gemm",
  "a_gemm",
  "c_gemm",
  "i_gemm",
  "o_gemm",
  "diq_gemm",
  "dik_gemm",
  "div_gemm",
  //"dia_gemm",
  //"dic_gemm",
  "dica_gemm",
  "dii_gemm",
  "dio_gemm",
  "dwqkv_gemm",
  //"dwq_gemm",
  //"dwk_gemm",
  //"dwv_gemm",
  "dwa_gemm",
  "dwc_gemm",
  "dwi_gemm",
  "dwo_gemm",
  "dqkv_bias",
  "di_bias",
  "do_bias",
  "w_vnni",
  "a_vnni",
  "w_xpose",
  "a_xpose",
  "other",
  "NUM_SCOPES"
};

extern PassType globalPass;
extern ScopeType globalScope;
constexpr int NUM_TIMERS = ((LAST_TIMER+7)/8)*8;
constexpr int NUM_ALIGNED_SCOPES = ((NUM_SCOPES+7)/8)*8;
extern double debug_timers[MAX_THREADS][2][NUM_TIMERS];
extern double scope_timers[MAX_THREADS][NUM_SCOPES][NUM_TIMERS];
extern double master_scope_timers[NUM_SCOPES];
extern double scope_flops[MAX_THREADS][NUM_ALIGNED_SCOPES];
class ScopedTimer {
  public:
  ScopedTimer(DebugTimer t, PassType p, long f=0) : type(t), pass(p), flops(f), start(getTime()) {}
  ~ScopedTimer() {
    auto time = getTime() - start;
    int tid = omp_get_thread_num();
    debug_timers[tid][pass][type] += time;
    scope_timers[tid][globalScope][type] += time;
    if (type == BRGEMM) scope_flops[tid][globalScope] += flops;
  }
  DebugTimer type;
  PassType pass;
  long flops;
  double start;
};

class GlobalScope {
  public:
  GlobalScope(ScopeType t) : oldScope(globalScope), start(getTime()) { globalScope = t; }
  ~GlobalScope() {
    auto time = getTime() - start;
    master_scope_timers[globalScope] += time;
    globalScope = oldScope;
  }
  ScopeType oldScope;
  double start;
};

extern double master_debug_timers[2];
class MasterScopedTimer {
  public:
  MasterScopedTimer(PassType p) : pass(p), start(getTime()) {}
  ~MasterScopedTimer() {
    auto time = getTime() - start;
    master_debug_timers[pass] += time;
  }
  PassType pass;
  double start;
};
#define PURE_GEMM_TIME
template<typename T, int impl=0>
class ScopedGEMMTPP {
  public:
  ScopedGEMMTPP(T func, DebugTimer t, long flops) : func(std::move(func)), t(t), flops(flops) { }
  template<typename Tin, typename Tout>
  void operator()(Tin *A, Tin *B, Tout *C, long count) {
#ifndef PURE_GEMM_TIME
    ScopedTimer _t(t, globalPass, 2*count*flops);
#endif
    if (impl == 0) {
      func(A, B, C, count);
    } else if (impl == 1) {
      func.ref(A, B, C, count);
    } else {
      printf("invalid impl requested\n");
      exit(1);
    }
  }
  private:
  T func;
  DebugTimer t;
  long flops;
};

template<typename T, int impl=0>
class ScopedTPP {
  public:
  ScopedTPP(T func, DebugTimer t) : func(std::move(func)), t(t) { }
  template<typename... Types>
  void operator()(Types... vars) {
    ScopedTimer _t(t, globalPass);
    if (impl == 0) {
      func(vars...);
    } else if (impl == 1) {
      func.ref(vars...);
    } else {
      printf("invalid impl requested\n");
      exit(1);
    }
  }
  private:
  T func;
  DebugTimer t;
};

#if 1
#define SCOPEITGEMM(f,t,flps) ScopedGEMMTPP<decltype(f)>(f, t, flps)
//#define SCOPEIT(f,t) ScopedTPP<decltype(f)>(f, t)
#define SCOPEIT(f,t) ScopedTPP<decltype(f),1>(f, t)
#define SCOPEIT_REF(f,t) ScopedTPP<decltype(f),1>(f, t)
#else
#define SCOPEIT(f,t) f
#endif

#define RECORD_SCOPE(scope,...) GlobalScope gs_(scope); RECORD_FUNCTION(#scope, std::vector<c10::IValue>(__VA_ARGS__))
#endif //_TIMING_H_
