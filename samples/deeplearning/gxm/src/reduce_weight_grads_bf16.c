int jobs = ofm * ifm * kh * kw;
assert(jobs % VLEN == 0);
int jn = jobs/gp->num_numa_nodes;
int jnv = jn/VLEN;
int jpt = (jnv % ntps == 0) ? (jnv/ntps)*VLEN: ((jnv/ntps)+1)*VLEN;
int ltid = tid - n*ntps;
int tb = (ltid * jpt < jn) ? ltid*jpt : jn;
int te = ((ltid+1)*jpt < jn) ? (ltid+1)*jpt : jn;

libxsmm_bfloat16 *my_ptr = (libxsmm_bfloat16*)dwt_ptr[n]+n*jn;

for(int nn=0; nn<gp->num_numa_nodes; nn++)
{
  if(n == nn) continue;

  libxsmm_bfloat16 *rem_ptr = (libxsmm_bfloat16*)dwt_ptr[nn]+n*jn;

  for(int i=tb; i<te; i+=VLEN)
  {
    __m256i vbfp16_l = _mm256_loadu_si256( (const __m256i*)(my_ptr+i));
    __m256i vbfp16_r = _mm256_loadu_si256( (const __m256i*)(rem_ptr+i));
    __m512  vfp32_l  = gxm_bfp16_to_fp32_avx512f( vbfp16_l );
    __m512  vfp32_r  = gxm_bfp16_to_fp32_avx512f( vbfp16_r );
    __m512  vfp32 = gxm_fp32_to_bfp16_rne_adjustment_avx512f( _mm512_add_ps(vfp32_l, vfp32_r));
    __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f( vfp32 );
    _mm256_storeu_si256( (__m256i*)(my_ptr+i), vbfp16 );
  }
}

#pragma omp barrier

for(int nn=0; nn<gp->num_numa_nodes; nn++)
{
  if(n == nn) continue;

  libxsmm_bfloat16 *my_ptr = (libxsmm_bfloat16*)dwt_ptr[n]+nn*jn;
  libxsmm_bfloat16 *rem_ptr = (libxsmm_bfloat16*)dwt_ptr[nn]+nn*jn;

#pragma omp simd
  for(int i=tb; i<te; i++)
    my_ptr[i] = rem_ptr[i];
}
