/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#include "fillers.hpp"

#define MEAN 0.0
#define NUM_SAMPLES 25
#define FAN_IN 0
#define FAN_OUT 1
#define AVERAGE 2

using namespace std;

void Uniform(const float lower, const float upper, int n, float *dist, unsigned int seed)
{
  for(int i=0; i<n; i++)
  {
    float random_number = (float)drand48();
    float retval = lower + (upper-lower)*random_number;
    dist[i] = retval;
  }
}

void Gaussian(float mean, float stddev, int n, float *dist, unsigned int seed)
{
  std::default_random_engine generator (seed);

  std::normal_distribution<double> distribution (mean, stddev);

  for(int i=0; i<n; i++)
    dist[i] = distribution(generator);
}

void initBuffer(void* ptr, int dtype, int vnorm, int fanin, int fanout, long long int bytes, string filler, unsigned int seed, float std)
{
  if(dtype == DT_FLOAT)
  {
    long long int size = bytes/sizeof(float);
    int divisor = fanin;
    if(vnorm == FAN_OUT)
      divisor = fanout;
    else if(vnorm == AVERAGE)
      divisor = (fanin + fanout)/2;

    if(filler.compare("msra") == 0)
      Gaussian(0.0, sqrt(2.0/(float)divisor), size, (float*)ptr, seed);
    else if(filler.compare("CAFFE") == 0)
      Gaussian(0.0, std, size, (float*)ptr, seed);
    else if(filler.compare("XAVIER") == 0)
      Uniform(-sqrt(3.0f/(float)divisor), sqrt(3.0f/(float)divisor), size, (float*)ptr, seed);
    else if(filler.compare("Gaussian") == 0)
      Gaussian(0.0, std, size, (float*)ptr, seed);
    else if(filler.compare("RANDOM01") == 0)
    {
      float *p = (float*)ptr;
      for(long long int i=0; i<size; i++)
        p[i] = drand48();
    }
  }
  else if(dtype == DT_INT16)
  {
    long long int size = bytes/sizeof(short int);

    short int*p = (short int*)ptr;
    if(filler.compare("ZERO") == 0)
      for(long long int i=0; i<size; i++)
        p[i] = 0;
    else
      printf("Weight filling for int16 not implemented!\n");
  }
}

void initConstantBuffer(void* ptr, int dtype, long long int bytes, string filler, float v)
{
  float *p = (float*)ptr;
  long long int size = bytes/sizeof(float);

  if(filler.compare("ONE") == 0)
    for(long long int i=0; i<size; i++)
      p[i] = 1.0f;
  else if(filler.compare("ZERO") == 0)
    for(long long int i=0; i<size; i++)
      p[i] = 0.0f;
  else if(filler.compare("CONSTANT") == 0)
    for(long long int i=0; i<size; i++)
      p[i] = v;
}

void initConstantBuffer(void* ptr, int dtype, long long int bytes, string filler, short int v)
{
  short int *p = (short int*)ptr;
  long long int size = bytes/sizeof(short int);

  if(filler.compare("ONE") == 0)
    for(long long int i=0; i<size; i++)
      p[i] = 1;
  else if(filler.compare("ZERO") == 0)
    for(long long int i=0; i<size; i++)
      p[i] = 0;
  else if(filler.compare("CONSTANT") == 0)
    for(long long int i=0; i<size; i++)
      p[i] = v;
}

