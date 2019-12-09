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


#pragma once
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include "Tensor.hpp"

using namespace std;

void Uniform(const float lower, const float upper, int n, float *ptr);
void Gaussian(float mean, float stddev, int n, float *ptr);
void initBuffer(void*, int vnorm, int fanin, int fanout, long long int, string, float std=0);
void initConstantBuffer(void*, long long int, string, float);
void initConstantBuffer(void*, long long int, string, short);
