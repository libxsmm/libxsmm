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

#ifndef PERF_COUNTER_MARKERS_H
#define PERF_COUNTER_MARKERS_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CTRS_CPU_SKX

#ifdef CTRS_CPU_SKX
#define CTRS_NCORE 28
#define CTRS_NIMC 6
#define CTRS_NCHA CTRS_NCORE
#endif

#ifndef CTRS_NCORE
#error no CTRS_CPU_[NAME] was specified
#endif

typedef enum ctrs_uncore_exp {
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
} ctrs_uncore_exp;

typedef struct ctrs_uncore {
  uint64_t act_rd[CTRS_NIMC];
  uint64_t act_wr[CTRS_NIMC];
  uint64_t cas_rd[CTRS_NIMC];
  uint64_t cas_wr[CTRS_NIMC];
  uint64_t imc_clockticks[CTRS_NIMC];
  uint64_t cha_rd[CTRS_NCHA];
  uint64_t cha_wr[CTRS_NCHA];
  uint64_t vert_bl_ring_in_use[CTRS_NCHA];
  uint64_t horz_bl_ring_in_use[CTRS_NCHA];
  uint64_t vert_ak_ring_in_use[CTRS_NCHA];
  uint64_t horz_ak_ring_in_use[CTRS_NCHA];
  uint64_t vert_iv_ring_in_use[CTRS_NCHA];
  uint64_t horz_iv_ring_in_use[CTRS_NCHA];
  uint64_t vert_txr_cycle_full[CTRS_NCHA];
  uint64_t horz_txr_cycle_full[CTRS_NCHA];
  uint64_t llc_lookup_rd[CTRS_NCHA];
  uint64_t llc_lookup_wr[CTRS_NCHA];
  uint64_t llc_victims[CTRS_NCHA];
  uint64_t xsnp_resp[CTRS_NCHA];
  uint64_t core_snp[CTRS_NCHA];
  uint64_t snoops_sent[CTRS_NCHA];
  uint64_t snoop_resp[CTRS_NCHA];
  uint64_t snoop_resp_local[CTRS_NCHA];
  uint64_t osb[CTRS_NCHA];
  uint64_t tor_inserts[CTRS_NCHA];
  uint64_t tor_occupancy[CTRS_NCHA];
  uint64_t cha_clockticks[CTRS_NCHA];
  uint64_t cms_clockticks[CTRS_NCHA];
  ctrs_uncore_exp exp;
} ctrs_uncore;

typedef enum ctrs_core_exp {
  CTRS_EXP_L2_BW,
  CTRS_EXP_CORE_SNP_RSP
} ctrs_core_exp;

typedef struct ctrs_core {
  uint64_t clockticks[CTRS_NCORE];
  uint64_t l2_lines_in[CTRS_NCORE];
  uint64_t l2_lines_out_s[CTRS_NCORE];
  uint64_t l2_lines_out_ns[CTRS_NCORE];
  uint64_t idi_misc_wb_up[CTRS_NCORE];
  uint64_t idi_misc_wb_down[CTRS_NCORE];
  uint64_t core_snp_rsp_ihiti[CTRS_NCORE];
  uint64_t core_snp_rsp_ihitfse[CTRS_NCORE];
  uint64_t core_snp_rsp_ifwdm[CTRS_NCORE];
  uint64_t core_snp_rsp_ifwdfe[CTRS_NCORE];
  ctrs_core_exp exp;
} ctrs_core;

typedef struct bw_gibs {
  double rd;
  double rd2;
  double wr;
  double wr2;
  double wr3;
  double wr4;
} bw_gibs;

typedef struct bw_bc {
  double cyc;
  double rd;
  double rd2;
  double wr;
  double wr2;
  double wr3;
  double wr4;
} bw_bc;

typedef struct snp_rsp {
  double cyc;
  double ihiti;
  double ihitfse;
  double ifwdm;
  double ifwdfe;
} snp_rsp;

void setup_uncore_ctrs( ctrs_uncore_exp exp );
void read_uncore_ctrs( ctrs_uncore *c );
void zero_uncore_ctrs( ctrs_uncore *c );
void divi_uncore_ctrs( ctrs_uncore *c, uint64_t div );
void difa_uncore_ctrs( const ctrs_uncore *a, const ctrs_uncore *b, ctrs_uncore* c );
void get_act_ddr_bw_uncore_ctrs( const ctrs_uncore *c, const double t, bw_gibs* bw );
void get_cas_ddr_bw_uncore_ctrs( const ctrs_uncore *c, const double t, bw_gibs* bw );
void get_llc_bw_uncore_ctrs( const ctrs_uncore *c, const double t, bw_gibs* bw );

void setup_core_ctrs( ctrs_core_exp exp );
void read_core_ctrs( ctrs_core *c );
void zero_core_ctrs( ctrs_core *c );
void divi_core_ctrs( ctrs_core *c, uint64_t div );
void difa_core_ctrs( const ctrs_core *a, const ctrs_core *b, ctrs_core* c );
void get_l2_bw_core_ctrs( const ctrs_core *c, const double t, bw_gibs* bw );
void get_l2_bytecycle_core_ctrs( const ctrs_core *c, bw_bc* bw );
void get_snp_rsp_core_ctrs( const ctrs_core *c, snp_rsp* rsp );

#ifdef __cplusplus
}
#endif

#endif /* PERF_COUNTER_MARKERS_H */

