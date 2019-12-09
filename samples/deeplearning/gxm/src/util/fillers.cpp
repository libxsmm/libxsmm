/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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

void Uniform(const float lower, const float upper, int n, float *dist)
{
  /*
  for(int i=0; i<n; i++)
  {
    float random_number = (float)drand48();
    float retval = lower + (upper-lower)*random_number;
    dist[i] = retval;
  }
  */
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::uniform_real_distribution<double> distribution(lower,upper);

  for(int i=0; i<n; i++)
    dist[i] = distribution(generator);
}

void Gaussian(float mean, float stddev, int n, float *dist)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);

  std::normal_distribution<double> distribution (mean, stddev);

  for(int i=0; i<n; i++)
    dist[i] = distribution(generator);
}

void initBuffer(void* ptr, int vnorm, int fanin, int fanout, long long int bytes, string filler, float std)
{
  long long int size = bytes/sizeof(float);
  int divisor = fanin;
  if(vnorm == FAN_OUT)
    divisor = fanout;
  else if(vnorm == AVERAGE)
    divisor = (fanin + fanout)/2;

  if(filler.compare("msra") == 0)
    Gaussian(0.0, sqrt(2.0/(float)divisor), size, (float*)ptr);
  else if(filler.compare("CAFFE") == 0)
    Gaussian(0.0, std, size, (float*)ptr);
  else if(filler.compare("XAVIER") == 0)
    Uniform(-sqrt(3.0f/(float)divisor), sqrt(3.0f/(float)divisor), size, (float*)ptr);
  else if(filler.compare("Gaussian") == 0)
    Gaussian(0.0, std, size, (float*)ptr);
  else if(filler.compare("RANDOM01") == 0)
  {
    float *p = (float*)ptr;
    for(long long int i=0; i<size; i++)
      p[i] = drand48();
  }
}

void initConstantBuffer(void* ptr, long long int bytes, string filler, float v)
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

void initConstantBuffer(void* ptr, long long int bytes, string filler, short int v)
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

