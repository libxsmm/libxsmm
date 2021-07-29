/******************************************************************************
** Copyright (c) 2020-2020, Intel Corporation                                **
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
/* Sanchit Misra (Intel Corp), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef COUNTERS_SKX_H
#define COUNTERS_SKX_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SKX_NCORE 28
#define SKX_NIMC 6
#define SKX_NCHA SKX_NCORE

typedef enum ctrs_skx_uc_exp {
  CTRS_EXP_DRAM_ACT,
  CTRS_EXP_DRAM_CAS,
  CTRS_EXP_CHA_ACT,
  CTRS_EXP_CHA_LLC_LOOKUP_VICTIMS,
  CTRS_EXP_CHA_XSNP_RESP,
  CTRS_EXP_CHA_CORE_SNP,
  CTRS_EXP_CHA_SNOOPS_SENT,
  CTRS_EXP_CHA_SNOOP_RESP_ALL,
  CTRS_EXP_CHA_OSB,
  CTRS_EXP_CHA_TOR,
  CTRS_EXP_CMS_BL,
  CTRS_EXP_CMS_AK,
  CTRS_EXP_CMS_IV,
  CTRS_EXP_CMS_AK_IV,
  CTRS_EXP_CMS_TXR_CYCLES_FULL
} ctrs_skx_uc_exp;

typedef struct ctrs_skx_uc {
  uint64_t act_rd[SKX_NIMC];
  uint64_t act_wr[SKX_NIMC];
  uint64_t cas_rd[SKX_NIMC];
  uint64_t cas_wr[SKX_NIMC];
  uint64_t imc_clockticks[SKX_NIMC];
  uint64_t cha_rd[SKX_NCHA];
  uint64_t cha_wr[SKX_NCHA];
  uint64_t vert_bl_ring_in_use[SKX_NCHA];
  uint64_t horz_bl_ring_in_use[SKX_NCHA];
  uint64_t vert_ak_ring_in_use[SKX_NCHA];
  uint64_t horz_ak_ring_in_use[SKX_NCHA];
  uint64_t vert_iv_ring_in_use[SKX_NCHA];
  uint64_t horz_iv_ring_in_use[SKX_NCHA];
  uint64_t vert_txr_cycle_full[SKX_NCHA];
  uint64_t horz_txr_cycle_full[SKX_NCHA];
  uint64_t llc_lookup_rd[SKX_NCHA];
  uint64_t llc_lookup_wr[SKX_NCHA];
  uint64_t llc_victims[SKX_NCHA];
  uint64_t xsnp_resp[SKX_NCHA];
  uint64_t core_snp[SKX_NCHA];
  uint64_t snoops_sent[SKX_NCHA];
  uint64_t snoop_resp[SKX_NCHA];
  uint64_t snoop_resp_local[SKX_NCHA];
  uint64_t osb[SKX_NCHA];
  uint64_t tor_inserts[SKX_NCHA];
  uint64_t tor_occupancy[SKX_NCHA];
  uint64_t cha_clockticks[SKX_NCHA];
  uint64_t cms_clockticks[SKX_NCHA];
  ctrs_skx_uc_exp exp;
} ctrs_skx_uc;

typedef enum ctrs_skx_core_exp {
  CTRS_EXP_L2_BW
} ctrs_skx_core_exp;

typedef struct ctrs_skx_core {
  uint64_t l2_lines_in[SKX_NCORE];
  uint64_t l2_lines_out_ns[SKX_NCORE];
  uint64_t idi_misc_wb_up[SKX_NCORE];
  uint64_t idi_misc_wb_down[SKX_NCORE];
  ctrs_skx_core_exp exp;
} ctrs_skx_core;

typedef struct bw_gibs {
  double rd;
  double rd2;
  double wr;
  double wr2;
  double wr3;
} bw_gibs;

void setup_skx_uc_ctrs( ctrs_skx_uc_exp exp );
void read_skx_uc_ctrs( ctrs_skx_uc *c );
void zero_skx_uc_ctrs( ctrs_skx_uc *c );
void divi_skx_uc_ctrs( ctrs_skx_uc *c, uint64_t div );
void difa_skx_uc_ctrs( const ctrs_skx_uc *a, const ctrs_skx_uc *b, ctrs_skx_uc* c );
void get_act_ddr_bw_skx( const ctrs_skx_uc *c, const double t, bw_gibs* bw );
void get_cas_ddr_bw_skx( const ctrs_skx_uc *c, const double t, bw_gibs* bw );
void get_llc_bw_skx( const ctrs_skx_uc *c, const double t, bw_gibs* bw );

void setup_skx_core_ctrs( ctrs_skx_core_exp exp );
void read_skx_core_ctrs( ctrs_skx_core *c );
void zero_skx_core_ctrs( ctrs_skx_core *c );
void divi_skx_core_ctrs( ctrs_skx_core *c, uint64_t div );
void difa_skx_core_ctrs( const ctrs_skx_core *a, const ctrs_skx_core *b, ctrs_skx_core* c );
void get_l2_bw_skx( const ctrs_skx_core *c, const double t, bw_gibs* bw );

#ifdef __cplusplus
}
#endif

#endif /* COUNTERS_SKX_H */

