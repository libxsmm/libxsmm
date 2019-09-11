if(n == 0)
{
  int jobs = ofm * ifm * kh * kw;
  assert(jobs % VLEN == 0);
  int jv = jobs/VLEN;
  int rem = jv % ntps;
  int jpt = (rem == 0) ? (jv/ntps)*VLEN : ((jv-rem)/ntps)*VLEN;
  int tb = (tid * jpt < jobs) ? tid*jpt : jobs;
  int te = ((tid+1)*jpt < jobs) ? (tid+1)*jpt : jobs;

  libxsmm_bfloat16 *my_ptr = (libxsmm_bfloat16*)dwt_ptr[n];

  for(int nn=1; nn<gp->num_numa_nodes; nn++)
  {
    libxsmm_bfloat16 *rem_ptr = (libxsmm_bfloat16*)dwt_ptr[nn];

    for(int i=tb; i<te; i+=VLEN)
    {
      __m512  vfp32_l  = gxm_bfp16_to_fp32_avx512f(_mm256_loadu_si256( (const __m256i*)(my_ptr + i)));
      __m512  vfp32_r  = gxm_bfp16_to_fp32_avx512f(_mm256_loadu_si256( (const __m256i*)(rem_ptr + i)));
      __m512  vfp32 = _mm512_add_ps(vfp32_l, vfp32_r);
      __m512  vfp32rne = gxm_fp32_to_bfp16_rne_adjustment_avx512f(vfp32);
      __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f(vfp32rne);
      _mm256_storeu_si256( (__m256i*)(my_ptr + i), vbfp16);
    }

    //Remainder processing
    if(tid == 0)
    {
      if(rem > 0)
      {
        for(int i=ntps*jpt; i<jobs; i+=VLEN)
        {
          __m512  vfp32_l  = gxm_bfp16_to_fp32_avx512f(_mm256_loadu_si256( (const __m256i*)(my_ptr + i)));
          __m512  vfp32_r  = gxm_bfp16_to_fp32_avx512f(_mm256_loadu_si256( (const __m256i*)(rem_ptr + i)));
          __m512  vfp32 = _mm512_add_ps(vfp32_l, vfp32_r);
          __m512  vfp32rne = gxm_fp32_to_bfp16_rne_adjustment_avx512f(vfp32);
          __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f(vfp32rne);
          _mm256_storeu_si256( (__m256i*)(my_ptr + i), vbfp16);
        }
      }
    }
  }
}

