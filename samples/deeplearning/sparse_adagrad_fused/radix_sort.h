#ifndef _PCL_RADIX_SORT_
#define _PCL_RADIX_SORT_

#include <utility>
#include <omp.h>

template<typename T>
using Key_Value_Pair = std::pair<T, T>;

  template<typename T>
Key_Value_Pair<T>* radix_sort_parallel(Key_Value_Pair<T>* inp_buf, Key_Value_Pair<T>* tmp_buf, int64_t elements_count, int64_t max_value)
{
  int maxthreads = omp_get_max_threads();
  int histogram[256*maxthreads], histogram_ps[256*maxthreads + 1];
  if(max_value == 0) return inp_buf;
  int num_bits = sizeof(T) * 8 - __builtin_clz(max_value);
  int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int * local_histogram = &histogram[256*tid];
    int * local_histogram_ps = &histogram_ps[256*tid];
    int elements_count_4 = elements_count/4*4;
    Key_Value_Pair<T> * input = inp_buf;
    Key_Value_Pair<T> * output = tmp_buf;

    for(unsigned int pass = 0; pass < num_passes; pass++)
    {

      /* Step 1: compute histogram
         Reset histogram */
      for(int i = 0; i < 256; i++) local_histogram[i] = 0;

#pragma omp for schedule(static)
      for(int64_t i = 0; i < elements_count_4; i+=4)
      {
        T val_1 = input[i].first;
        T val_2 = input[i+1].first;
        T val_3 = input[i+2].first;
        T val_4 = input[i+3].first;

        local_histogram[ (val_1>>(pass*8)) &0xFF]++;
        local_histogram[ (val_2>>(pass*8)) &0xFF]++;
        local_histogram[ (val_3>>(pass*8)) &0xFF]++;
        local_histogram[ (val_4>>(pass*8)) &0xFF]++;
      }
      if(tid == (nthreads -1))
      {
        for(int64_t i = elements_count_4; i < elements_count; i++)
        {
          T val = input[i].first;
          local_histogram[ (val>>(pass*8)) &0xFF]++;
        }
      }
#pragma omp barrier
      /* Step 2: prefix sum */
      if(tid == 0)
      {
        int sum = 0, prev_sum = 0;
        for(int bins = 0; bins < 256; bins++) for(int t = 0; t < nthreads; t++) { sum += histogram[t*256 + bins]; histogram_ps[t*256 + bins] = prev_sum; prev_sum = sum; }
        histogram_ps[256*nthreads] = prev_sum; if(prev_sum != elements_count) { printf("Error1!\n"); exit(123); }
      }
#pragma omp barrier

      /* Step 3: scatter */
#pragma omp for schedule(static)
      for(int64_t i = 0; i < elements_count_4; i+=4)
      {
        T val_1 = input[i].first;
        T val_2 = input[i+1].first;
        T val_3 = input[i+2].first;
        T val_4 = input[i+3].first;
        T bin_1 = (val_1>>(pass*8)) &0xFF;
        T bin_2 = (val_2>>(pass*8)) &0xFF;
        T bin_3 = (val_3>>(pass*8)) &0xFF;
        T bin_4 = (val_4>>(pass*8)) &0xFF;
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
      if(tid == (nthreads -1))
      {
        for(int64_t i = elements_count_4; i < elements_count; i++)
        {
          T val = input[i].first;
          int pos = local_histogram_ps[ (val>>(pass*8)) &0xFF]++;
          output[pos] = input[i];
        }
      }

      Key_Value_Pair<T> * temp = input; input = output; output = temp;
#pragma omp barrier
    }
  }
  return (num_passes % 2 == 0 ? inp_buf : tmp_buf);
}

#endif

