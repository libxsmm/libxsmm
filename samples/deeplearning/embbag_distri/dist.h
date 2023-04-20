/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/
#ifndef _DIST_H_
#define _DIST_H_

#ifdef USE_MPI
#include <mpi.h>
void dist_init(int*argc, char ***argv)
{
  MPI_Init(argc, argv);
}

void dist_fini()
{
  MPI_Finalize();
}

int dist_get_rank()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int dist_get_size()
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

void dist_barrier()
{
  MPI_Barrier(MPI_COMM_WORLD);
}

void dist_alltoall(int count, float* sendbuf, float*recvbuf)
{
  MPI_Alltoall(sendbuf, count, MPI_FLOAT, recvbuf, count, MPI_FLOAT, MPI_COMM_WORLD);
}
#elif defined(USE_CCL)
#include <ccl.hpp>
static ccl::communicator_t comm;
void dist_init(int*argc, char ***argv)
{
  comm = ccl::environment::instance().create_communicator();
}

void dist_fini()
{
  comm.reset();
}

int dist_get_rank()
{
  return comm->rank();
}

int dist_get_size()
{
  return comm->size();
}

void dist_barrier()
{
  comm->barrier();
}

void dist_alltoall(int count, float* sendbuf, float*recvbuf)
{
  comm->alltoall(sendbuf, recvbuf, (size_t)count, ccl::datatype::dt_float)->wait();
}
#else
void dist_init(int*argc, char ***argv)
{
  return;
}

void dist_fini()
{
  return;
}

int dist_get_rank()
{
  return 0;
}

int dist_get_size()
{
  return 1;
}

void dist_barrier()
{
  return;
}

void dist_alltoall(int count, float* sendbuf, float*recvbuf)
{
#pragma omp parallel for
  for (int i = 0; i < count; i++)
  {
      recvbuf[i] = sendbuf[i];
  }
}
#endif
#endif /* _DIST_H_ */
