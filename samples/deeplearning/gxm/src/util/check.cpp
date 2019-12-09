/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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
                float val = tensor_vla[img][fm][fhi-1-ph][w][v];
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

void check_physical_pad(const char *s, libxsmm_bfloat16 *tensor, int nImg, int nBfm, int fh, int fw, int ifm, int iph, int ipw ) {
  int fhi = fh + 2*iph;
  int fwi = fw + 2*ipw;
  bool success = true;
  bool padded = false;

  libxsmm_bfloat16 (* __restrict tensor_vla)[nBfm][fhi][fwi][ifm] = (libxsmm_bfloat16 (*)[*][*][*][ifm])tensor;

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

void MeanOfLayer(char *s, libxsmm_bfloat16 *array, int size)
{
  union libxsmm_bfloat16_hp max, min, sum, absum;

  max.i[0] = 0;
  max.i[1] = array[0];
  min.i[0] = 0;
  min.i[1] = array[0];
  sum.i[0] = 0;
  sum.i[1] = array[0];
  absum.i[0] = 0;
  absum.i[1] = (array[0] > 0) ? array[0] : -array[0];

  for(int i=1; i<size; i++)
  {
    union libxsmm_bfloat16_hp val;
    val.i[0] = 0;
    val.i[1] = array[i];

    if(val.f > max.f)
      max.f = val.f;

    if(val.f < min.f)
      min.f = val.f;

    sum.f += val.f;
    absum.f += fabs(val.f);
  }
  printf("%s %f %f %f %f\n", s, sum.f, absum.f, max.f, min.f);
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
  printf("%s %.10f %.10f %.10f %.10f\n", s, sum, absum, max, min);
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

    //printf("layer:%s(%d) mean:%f stddev=%f max:%f(%d) min:%f(%d) \n", layer, size,  mean, stddev, max, which_max, min, which_min);

//  printf("%s:[%d] mean:%.10f (abs mean:%.10f) stddev:%.10f (abs stdev %.10f) max:%.10f(%d) min:%.10f(%d) nnz-perc:%.10f(%d:f=%d l=%d) \n",
//      s, size,  mean, absmean, stddev, abstddev, max, which_max, min, which_min, ((double)nnz)/((double)size), nnz, first_nz, last_nz);
  printf("%s %d %d %d %d\n", s, sum, absum, max, min);
}
