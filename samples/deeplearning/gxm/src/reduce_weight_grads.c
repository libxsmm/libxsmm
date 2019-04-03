int jobs = ofm * ifm * kh * kw;
int jn = jobs/gp->num_numa_nodes;
int jnv = jn/VLEN;
int jpt = (jnv % ntps == 0) ? (jnv/ntps)*VLEN : ((jnv/ntps)+1)*VLEN;
int ltid = tid - n*ntps;
int tb = (ltid * jpt < jn) ? ltid*jpt : jn;
int te = ((ltid+1)*jpt < jn) ? (ltid+1)*jpt : jn;

float *wgp = (float*)dwt_ptr[n]+n*jn;

for(int nn=0; nn<gp->num_numa_nodes; nn++)
{
  if(n == nn) continue;

  float *rgp = (float*)dwt_ptr[nn]+n*jn;

#pragma omp simd
  for(int i=tb; i<te; i++)
    wgp[i] += rgp[i];
}

#pragma omp barrier

for(int nn=0; nn<gp->num_numa_nodes; nn++)
{
  if(n == nn) continue;
  float *wgp = (float*)dwt_ptr[n]+nn*jn;
  float *rgp = (float*)dwt_ptr[nn]+nn*jn;

#pragma vector nontemporal
#pragma omp simd
  for(int i=tb; i<te; i++)
    wgp[i] = rgp[i];
}
