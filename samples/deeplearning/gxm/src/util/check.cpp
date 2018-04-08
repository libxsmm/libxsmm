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
/* Sasikanth Avancha, Dhiraj Kalamkar, Alexander Heinecke  (Intel Corp.)
******************************************************************************/

#include "check.hpp"

void check_physical_pad(const char *s, float *tensor, int nImg, int nBfm, int fh, int fw, int ifm, int iph, int ipw ) {
  int fhi = fh + 2*iph;
  int fwi = fw + 2*ipw;
  bool success = true;
  bool padded = false;

  float (* __restrict tensor_vla)[nBfm][fhi][fwi][ifm] = (float (*)[*][*][*][ifm])tensor;

  if (iph > 0 || iph > 0) {
    for (int img = 0; img < nImg; img++) {
      for (int fm = 0; fm < nBfm; fm++) {
        for (int w = 0; w < fwi; w++) {
          for (int ph = 0; ph < iph; ph++) {
            for (int v = 0; v < ifm; v++) {
              if ( tensor_vla[img][fm][ph][w][v] != 0.0f ) {
                success = false;
              }
              if ( tensor_vla[img][fm][fhi-1-ph][w][v] != 0.0f ) {
                success = false;
              }
            }
          }
        }
        for (int h = iph; h < fh+iph; h++) {
          for (int pw = 0; pw < ipw; pw++) {
            for (int v = 0; v < ifm; v++) {
              if ( tensor_vla[img][fm][h][pw][v] != 0.0f ) {
                success = false;
              }
              if ( tensor_vla[img][fm][h][fwi-1-pw][v] != 0.0f ) {
                success = false;
              }
            }
          }
        }
      }
    }
    padded = true;
  }

  if ( padded == true ) {
    if ( success == true ) {
      printf("%s pacific_rim is clear\n", s);
    } else {
      printf("%s pacific_rim is under attack\n", s);
    }
  }
}

void check_physical_pad(const char *s, short *tensor, int nImg, int nBfm, int fh, int fw, int ifm, int iph, int ipw ) {
  int fhi = fh + 2*iph;
  int fwi = fw + 2*ipw;
  bool success = true;
  bool padded = false;

  short (* __restrict tensor_vla)[nBfm][fhi][fwi][ifm] = (short (*)[*][*][*][ifm])tensor;

  if (iph > 0 || iph > 0) {
    for (int img = 0; img < nImg; img++) {
      for (int fm = 0; fm < nBfm; fm++) {
        for (int w = 0; w < fwi; w++) {
          for (int ph = 0; ph < iph; ph++) {
            for (int v = 0; v < ifm; v++) {
              if ( tensor_vla[img][fm][ph][w][v] != 0 ) {
                success = false;
              }
              if ( tensor_vla[img][fm][fhi-1-ph][w][v] != 0 ) {
                success = false;
              }
            }
          }
        }
        for (int h = iph; h < fh+iph; h++) {
          for (int pw = 0; pw < ipw; pw++) {
            for (int v = 0; v < ifm; v++) {
              if ( tensor_vla[img][fm][h][pw][v] != 0 ) {
                success = false;
              }
              if ( tensor_vla[img][fm][h][fwi-1-pw][v] != 0 ) {
                success = false;
              }
            }
          }
        }
      }
    }
    padded = true;
  }

  if ( padded == true ) {
    if ( success == true ) {
      printf("%s pacific_rim is clear\n", s);
    } else {
      printf("%s pacific_rim is under attack\n", s);
    }
  }
}
void MeanOfLayer(char *s, short *array, int size)
{
  int nnz, mmt;
  short max, min;
  long int sum, absum;

  max = array[0];
  min = array[0];
  sum = 0;
  absum = 0;
  nnz = 0;
  mmt = 0;

  int which_max = 0;
  int which_min = 0;
  int first_nz = -1;
  int last_nz = -1;

  for(int i=0; i<size; i++)
  {
    if(array[i] != 0) last_nz = i;
    if(first_nz == -1 && array[i] != 0) first_nz = i;
#if 0
    if(array[i] > 1000 || array[i] < -1000) {mmt++; printf(">>%d (%f)\n", i, array[i]);}
    if(mmt > 10)
    {
      printf("In %s more than 10 values out-of-range. exiting statistics loop...\n",s);
      exit(0);
    }
#endif
    if(array[i] != 0) nnz++;
    if(array[i] > max)
    {
      max = array[i];
      which_max = i;
    }
    if(array[i] < min)
    {
      min = array[i];
      which_min = i;
    }
    sum += array[i];

    absum += fabs(array[i]);
  }
  printf("%s %ld %ld %d %d\n", s, sum, absum, max, min);
}

void MeanOfLayer(char *s, float *array, int size)
{

  int nnz, mmt;
  double max, min;
  double sum, absum;
  double stddev_sum, stddev_absum;

  max = array[0];
  min = array[0];
  sum = 0;
  double psum=0, nsum=0;
  int pc=0, nc=0;
  absum = 0;
  nnz = 0;
  mmt = 0;

  int which_max = 0;
  int which_min = 0;
  int first_nz = -1;
  int last_nz = -1;

  for(int i=0; i<size; i++)
  {
    if(array[i] != 0) last_nz = i;
    if(first_nz == -1 && array[i] != 0) first_nz = i;
#if 0
    if(array[i] > 1000 || array[i] < -1000) {mmt++; printf(">>%d (%f)\n", i, array[i]);}
    if(mmt > 10)
    {
      printf("In %s more than 10 values out-of-range. exiting statistics loop...\n",s);
      exit(0);
    }
#endif
    if(array[i] != 0) nnz++;
    if(array[i] > max)
    {
      max = array[i];
      which_max = i;
    }
    if(array[i] < min)
    {
      min = array[i];
      which_min = i;
    }
    sum += array[i];
    if(array[i] > 0)
    {
      psum += array[i];
      pc++;
    }
    else if(array[i] < 0)
    {
      nsum += array[i];
      nc++;
    }

    absum += fabs(array[i]);
  }
  double mean = sum/(double)size;
  double absmean = absum/(double)size;

  stddev_sum = 0;
  for(int i=0; i<size; i++)
  {
    stddev_sum += (array[i] - mean)*(array[i] - mean);
  }

  stddev_sum = stddev_sum/size;

  double stddev = sqrt(stddev_sum);

  //  printf("layer:%s(%d) mean:%f stddev=%f max:%f(%d) min:%f(%d) \n", layer, size,  mean, stddev, max, which_max, min, which_min);

//  printf("%s:[%d] mean:%.10f (abs mean:%.10f) stddev:%.10f (abs stdev %.10f) max:%.10f(%d) min:%.10f(%d) nnz-perc:%.10f(%d:f=%d l=%d) \n",
//      s, size,  mean, absmean, stddev, abstddev, max, which_max, min, which_min, ((double)nnz)/((double)size), nnz, first_nz, last_nz);
  //printf("%s %.10f %.10f %.10f %.10f\n", s, mean, stddev, max, min);
  //printf("%s %10g %10g %10g %10g(%d) %10g(%d), nnz-perc:%.10g(%d:f=%d l=%d)\n", s, mean, absmean, stddev, max, which_max, min, which_min, ((double)nnz)/((double)size), nnz, first_nz, last_nz);
  printf("%s %10g %10g %10g %10g\n", s, sum, absum, max, min);
}

void MeanOfLayer(char *s, double *array, int size)
{

  int nnz, mmt;
  double max, min;
  double sum, absum;
  double stddev_sum, stddev_absum;

  max = array[0];
  min = array[0];
  sum = 0;
  double psum=0, nsum=0;
  int pc=0, nc=0;
  absum = 0;
  nnz = 0;
  mmt = 0;

  int which_max = 0;
  int which_min = 0;
  int first_nz = -1;
  int last_nz = -1;

  for(int i=0; i<size; i++)
  {
    if(array[i] != 0) last_nz = i;
    if(first_nz == -1 && array[i] != 0) first_nz = i;
#if 0
    if(array[i] > 1000 || array[i] < -1000) {mmt++; printf(">>%d (%f)\n", i, array[i]);}
    if(mmt > 10)
    {
      printf("In %s more than 10 values out-of-range. exiting statistics loop...\n",s);
      exit(0);
    }
#endif
    if(array[i] != 0) nnz++;
    if(array[i] > max)
    {
      max = array[i];
      which_max = i;
    }
    if(array[i] < min)
    {
      min = array[i];
      which_min = i;
    }
    sum += array[i];
    if(array[i] > 0)
    {
      psum += array[i];
      pc++;
    }
    else if(array[i] < 0)
    {
      nsum += array[i];
      nc++;
    }

    absum += fabs(array[i]);
  }
  double mean = sum/(double)size;
  double absmean = absum/(double)size;

  stddev_sum = 0;
  for(int i=0; i<size; i++)
  {
    stddev_sum += (array[i] - mean)*(array[i] - mean);
  }

  stddev_sum = stddev_sum/size;

  double stddev = sqrt(stddev_sum);

  //  printf("layer:%s(%d) mean:%f stddev=%f max:%f(%d) min:%f(%d) \n", layer, size,  mean, stddev, max, which_max, min, which_min);

//  printf("%s:[%d] mean:%.10f (abs mean:%.10f) stddev:%.10f (abs stdev %.10f) max:%.10f(%d) min:%.10f(%d) nnz-perc:%.10f(%d:f=%d l=%d) \n",
//      s, size,  mean, absmean, stddev, abstddev, max, which_max, min, which_min, ((double)nnz)/((double)size), nnz, first_nz, last_nz);
  //printf("%s %.10f %.10f %.10f %.10f\n", s, mean, stddev, max, min);
//  printf("%s %10g %10g %10g %10g(%d) %10g(%d), nnz-perc:%.10g(%d:f=%d l=%d)\n", s, mean, absmean, stddev, max, which_max, min, which_min, ((double)nnz)/((double)size), nnz, first_nz, last_nz);
  printf("%s %10g %10g %10g %10g\n", s, sum, absum, max, min);
}

void MeanOfLayer(char *s, int *array, int size)
{

  int nnz, mmt;
  int max, min;
  int sum, absum;
  double stddev_sum, stddev_absum;

  max = array[0];
  min = array[0];
  sum = 0;
  absum = 0;
  nnz = 0;
  mmt = 0;

  int which_max = 0;
  int which_min = 0;
  int first_nz = -1;
  int last_nz = -1;

  for(int i=0; i<size; i++)
  {
    if(array[i] != 0) last_nz = i;
    if(first_nz == -1 && array[i] != 0) first_nz = i;
    if(array[i] > 1000 || array[i] < -1000) {mmt++; printf(">>%d (%d)\n", i, array[i]);}
    if(mmt > 100)
    {
      printf("more than 100 values out-of-range. exiting statistics loop...\n");
      break;
    }
    if(array[i] != 0) nnz++;
    if(array[i] > max) {max = array[i]; which_max = i;}
    if(array[i] < min) {min = array[i]; which_min = i;}
    sum += array[i];
    absum += fabs(array[i]);
  }

  double mean = sum/size;
  double absmean = absum/size;

  stddev_sum = 0;
  stddev_absum = 0;
  for(int i=0; i<size; i++)
  {
    stddev_sum += (array[i] - mean)*(array[i] - mean);
    stddev_absum += (array[i] - absmean)*(array[i] - absmean);
  }

  stddev_sum = stddev_sum/size;
  stddev_absum = stddev_absum/size;

  double stddev = sqrt(stddev_sum);
  double abstddev = sqrt(stddev_absum);

    //printf("layer:%s(%d) mean:%f stddev=%f max:%f(%d) min:%f(%d) \n", layer, size,  mean, stddev, max, which_max, min, which_min);

//  printf("%s:[%d] mean:%.10f (abs mean:%.10f) stddev:%.10f (abs stdev %.10f) max:%.10f(%d) min:%.10f(%d) nnz-perc:%.10f(%d:f=%d l=%d) \n",
//      s, size,  mean, absmean, stddev, abstddev, max, which_max, min, which_min, ((double)nnz)/((double)size), nnz, first_nz, last_nz);
  printf("%s %.10f %.10f %d %d\n", s, mean, stddev, max, min);
}
