/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Sanchit Misra, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef COUNTERS_H
#define COUNTERS_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SKX_NIMC 6
#define SKX_NCHA 28

typedef enum ctrs_skx_uc_exp {
  CTRS_EXP_DRAM_ACT,
  CTRS_EXP_DRAM_CAS,
  CTRS_EXP_CHA_ACT,
  CTRS_EXP_CHA_BL_VERT,
  CTRS_EXP_CHA_BL_HORZ,
  CTRS_EXP_CHA_LLC_LOOKUP
} ctrs_skx_uc_exp;

typedef struct ctrs_skx_uc
{
  uint64_t act_rd[SKX_NIMC];
  uint64_t act_wr[SKX_NIMC];
  uint64_t cas_rd[SKX_NIMC];
  uint64_t cas_wr[SKX_NIMC];
  uint64_t imc_clockticks[SKX_NIMC];
  uint64_t cha_rd[SKX_NCHA];
  uint64_t cha_wr[SKX_NCHA];
  uint64_t vert_ring_bl_in_use_up[SKX_NCHA];
  uint64_t vert_ring_bl_in_use_dn[SKX_NCHA];
  uint64_t horz_ring_bl_in_use_lf[SKX_NCHA];
  uint64_t horz_ring_bl_in_use_rt[SKX_NCHA];
  uint64_t llc_lookup_rd[SKX_NCHA];
  uint64_t llc_lookup_wr[SKX_NCHA];
  uint64_t cha_clockticks[SKX_NCHA];
  ctrs_skx_uc_exp exp;
} ctrs_skx_uc;

typedef struct bw_gibs {
  double rd;
  double wr;
} bw_gibs;

void setup_skx_uc_ctrs( ctrs_skx_uc_exp exp );
void read_skx_uc_ctrs( ctrs_skx_uc *c );
void zero_skx_uc_ctrs( ctrs_skx_uc *c );
void divi_skx_uc_ctrs( ctrs_skx_uc *c, uint64_t div );
void difa_skx_uc_ctrs( const ctrs_skx_uc *a, const ctrs_skx_uc *b, ctrs_skx_uc* c );
void get_cas_ddr_bw_skx( const ctrs_skx_uc *c, const double t, bw_gibs* bw );
void get_llc_bw_skx( const ctrs_skx_uc *c, const double t, bw_gibs* bw );

#ifdef __cplusplus
}
#endif

#endif /* COUNTERS_H */
