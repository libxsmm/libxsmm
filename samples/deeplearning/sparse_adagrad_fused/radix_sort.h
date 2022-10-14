#ifndef _PCL_RADIX_SORT_
#define _PCL_RADIX_SORT_

#include <utility>
#include <limits>

#include "utils.h"

#ifndef BKT_BITS
#define BKT_BITS 12
#endif

template<typename T>
using Key_Value_Pair = std::pair<T, T>;

template<typename T>
Key_Value_Pair<T>* radix_sort_parallel(Key_Value_Pair<T>* inp_buf, Key_Value_Pair<T>* tmp_buf, int64_t elements_count, int64_t max_value)
{
  constexpr int bkt_bits = BKT_BITS;
  constexpr int nbkts = (1 << bkt_bits);
  constexpr int bkt_mask = (nbkts - 1);

  int maxthreads = omp_get_max_threads();
  int histogram[nbkts*maxthreads], histogram_ps[nbkts*maxthreads + 1];
  if (max_value == 0) return inp_buf;
  int num_bits = 64;
  if (sizeof(T) == 8 && max_value > std::numeric_limits<int>::max()) {
    num_bits = sizeof(T) * 8 - __builtin_clzll(max_value);
  } else {
    num_bits = 32 - __builtin_clz((unsigned int)max_value);
  }

  int num_passes = (num_bits + bkt_bits - 1) / bkt_bits;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int * local_histogram = &histogram[nbkts*tid];
    int * local_histogram_ps = &histogram_ps[nbkts*tid];
    int elements_count_4 = elements_count/4*4;
    Key_Value_Pair<T> * input = inp_buf;
    Key_Value_Pair<T> * output = tmp_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++)
    {

      auto t1 = get_time();
      /* Step 1: compute histogram
         Reset histogram */
      for (int i = 0; i < nbkts; i++) local_histogram[i] = 0;

#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i+=4)
      {
        T val_1 = input[i].first;
        T val_2 = input[i+1].first;
        T val_3 = input[i+2].first;
        T val_4 = input[i+3].first;

        local_histogram[ (val_1>>(pass*bkt_bits)) & bkt_mask]++;
        local_histogram[ (val_2>>(pass*bkt_bits)) & bkt_mask]++;
        local_histogram[ (val_3>>(pass*bkt_bits)) & bkt_mask]++;
        local_histogram[ (val_4>>(pass*bkt_bits)) & bkt_mask]++;
      }
      if (tid == (nthreads -1))
      {
        for (int64_t i = elements_count_4; i < elements_count; i++)
        {
          T val = input[i].first;
          local_histogram[ (val>>(pass*bkt_bits)) & bkt_mask]++;
        }
      }
#pragma omp barrier
      auto t11 = get_time();
      /* Step 2: prefix sum */
      if (tid == 0)
      {
        int sum = 0, prev_sum = 0;
        for (int bins = 0; bins < nbkts; bins++) for (int t = 0; t < nthreads; t++) { sum += histogram[t*nbkts + bins]; histogram_ps[t*nbkts + bins] = prev_sum; prev_sum = sum; }
        histogram_ps[nbkts*nthreads] = prev_sum; if (prev_sum != elements_count) { printf("Error1!\n"); exit(123); }
      }
#pragma omp barrier
      auto t12 = get_time();

      /* Step 3: scatter */
#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i+=4)
      {
        T val_1 = input[i].first;
        T val_2 = input[i+1].first;
        T val_3 = input[i+2].first;
        T val_4 = input[i+3].first;
        T bin_1 = (val_1>>(pass*bkt_bits)) & bkt_mask;
        T bin_2 = (val_2>>(pass*bkt_bits)) & bkt_mask;
        T bin_3 = (val_3>>(pass*bkt_bits)) & bkt_mask;
        T bin_4 = (val_4>>(pass*bkt_bits)) & bkt_mask;
        int pos;
        pos = local_histogram_ps[bin_1]++;
        output[pos] = input[i];
        pos = local_histogram_ps[bin_2]++;
        output[pos] = input[i+1];
        pos = local_histogram_ps[bin_3]++;
        output[pos] = input[i+2];
        pos = local_histogram_ps[bin_4]++;
        output[pos] = input[i+3];
      }
      if (tid == (nthreads -1))
      {
        for (int64_t i = elements_count_4; i < elements_count; i++)
        {
          T val = input[i].first;
          int pos = local_histogram_ps[ (val>>(pass*bkt_bits)) & bkt_mask]++;
          output[pos] = input[i];
        }
      }

      Key_Value_Pair<T> * temp = input; input = output; output = temp;
#pragma omp barrier
      auto t2 = get_time();
#ifdef DEBUG_TIME
      if (tid == 0) printf("pass = %d  total time = %.3f  step1 = %.3f  step2 = %.3f %.3f\n", pass, t2-t1, t11-t1, t12-t11, t2-t12);
#endif
    }
  }
  return (num_passes % 2 == 0 ? inp_buf : tmp_buf);
}

#endif

