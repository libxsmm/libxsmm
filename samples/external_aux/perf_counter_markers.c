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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <err.h>
#include <unistd.h>
#include <syscall.h>
#include <linux/perf_event.h>
#include <sys/types.h>
#include <sys/syscall.h>

#include "perf_counter_markers.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct perf_uncore_fd {
  int fd_act_rd[CTRS_NIMC];
  int fd_act_wr[CTRS_NIMC];
  int fd_cas_rd[CTRS_NIMC];
  int fd_cas_wr[CTRS_NIMC];
  int fd_imc_clockticks[CTRS_NIMC];
  int fd_cha_rd[CTRS_NCHA];
  int fd_cha_wr[CTRS_NCHA];
  int fd_vert_bl_ring_in_use[CTRS_NCHA];
  int fd_horz_bl_ring_in_use[CTRS_NCHA];
  int fd_vert_ak_ring_in_use[CTRS_NCHA];
  int fd_horz_ak_ring_in_use[CTRS_NCHA];
  int fd_vert_iv_ring_in_use[CTRS_NCHA];
  int fd_horz_iv_ring_in_use[CTRS_NCHA];
  int fd_llc_lookup_rd[CTRS_NCHA];
  int fd_llc_lookup_wr[CTRS_NCHA];
  int fd_llc_victims[CTRS_NCHA];
  int fd_xsnp_resp[CTRS_NCHA];
  int fd_core_snp[CTRS_NCHA];
  int fd_snoops_sent[CTRS_NCHA];
  int fd_snoop_resp[CTRS_NCHA];
  int fd_snoop_resp_local[CTRS_NCHA];
  int fd_osb[CTRS_NCHA];
  int fd_tor_inserts[CTRS_NCHA];
  int fd_tor_occupancy[CTRS_NCHA];
  int fd_cha_clockticks[CTRS_NCHA];
  int fd_cms_clockticks[CTRS_NCHA];
  ctrs_uncore_exp exp;
} perf_uncore_fd;

typedef struct perf_core_fd
{
  int fd_clockticks[CTRS_NCORE];
  int fd_l2_lines_in[CTRS_NCORE];
  int fd_l2_lines_out_ns[CTRS_NCORE];
  int fd_idi_misc_wb_up[CTRS_NCORE];
  int fd_idi_misc_wb_down[CTRS_NCORE];
  ctrs_core_exp exp;
} perf_core_fd;

static perf_uncore_fd gbl_uncore_perf_fd;
static perf_core_fd gbl_core_perf_fd;

static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags);
static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                  group_fd, flags);
  return ret;
}

void evsetup(const char *ename, int *fd, unsigned int event, unsigned int umask, unsigned int filter0, unsigned int filter1, int core );
void evsetup(const char *ename, int *fd, unsigned int event, unsigned int umask, unsigned int filter0, unsigned int filter1, int core ) {
  char fname[1024];
#if 0
  snprintf(fname, sizeof(fname), "%s/type", ename);
#else
  sprintf(fname, "%s/type", ename);
#endif
  FILE *fp = fopen(fname, "r");
  if (fp == 0) {
    err(1, "open %s", fname);
  }
  int type;
  int ret = fscanf(fp, "%d", &type);
  uint64_t filter0_64;
  uint64_t filter1_64;
  int cpu;
  int pid;

  assert(ret == 1);
  fclose(fp);
#if 0
  printf("Using PMU type %d from %s\n", type, ename);
#endif

  struct perf_event_attr hw = {};
  hw.size = sizeof(hw);
  hw.type = type;
/*
  see /sys/devices/uncore_?/format/?
  Although are events we are using here are configured in the same way,
  we should read the format.

  hw.read_format = PERF_FORMAT_GROUP;
  unfortunately the below only works within a single PMU; might
  as well just read them one at a time

  on two socket system we would need to create a second set for the
  second socket
*/

  hw.config   = event | (umask << 8);
  filter0_64 = (uint64_t)filter0;
  filter1_64 = (uint64_t)filter1;
  filter0_64 |= filter1_64 << 32;
  hw.config1  = filter0_64;

/*
  printf("0x%llx\n", hw.config1 );
*/

  if ( core < 0 ) {
    cpu = 0;
    pid = -1;
  } else {
    cpu = core;
    pid = -1;
  }

  *fd = perf_event_open(&hw, pid, cpu, -1, 0);
  if (*fd == -1) {
    err(1, "CPU %d, box %s, event 0x%llu", cpu, ename, hw.config);
  }
}

void setup_uncore_ctrs( ctrs_uncore_exp exp ) {
  int ret;
  char fname[1024];
  int mc, cha;

  for ( mc = 0; mc < CTRS_NIMC; ++mc ) {
#if 0
    snprintf(fname, sizeof(fname), "/sys/devices/uncore_imc_%d",mc);
#else
    sprintf(fname, "/sys/devices/uncore_imc_%d", mc);
#endif
#ifdef CTRS_CPU_SKX 
    if ( exp == CTRS_EXP_DRAM_ACT ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_act_rd[mc], 0x01, 0x01, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_act_wr[mc], 0x01, 0x02, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_imc_clockticks[mc], 0x00, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_DRAM_CAS ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_cas_rd[mc], 0x04, 0x03, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cas_wr[mc], 0x04, 0x0C, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_imc_clockticks[mc], 0x00, 0x00, 0x00, 0x00, -1);
    } else {
      /* nothing */
    }
#endif
  }

  for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
#if 0
    snprintf(fname, sizeof(fname), "/sys/devices/uncore_cha_%d",cha);
#else
    sprintf(fname, "/sys/devices/uncore_cha_%d", cha);
#endif
#ifdef CTRS_CPU_SKX
    if ( exp == CTRS_EXP_CHA_ACT ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_rd[cha], 0x50, 0x03, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_wr[cha], 0x50, 0x0C, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CMS_BL ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_vert_bl_ring_in_use[cha], 0xAA, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_horz_bl_ring_in_use[cha], 0xAB, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cms_clockticks[cha], 0xc0, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CMS_AK ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_vert_ak_ring_in_use[cha], 0xA8, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_horz_ak_ring_in_use[cha], 0xA9, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cms_clockticks[cha], 0xc0, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CMS_IV ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_vert_iv_ring_in_use[cha], 0xAC, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_horz_iv_ring_in_use[cha], 0xAD, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cms_clockticks[cha], 0xc0, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CMS_AK_IV ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_vert_ak_ring_in_use[cha], 0xA8, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_horz_ak_ring_in_use[cha], 0xA9, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_vert_iv_ring_in_use[cha], 0xAC, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_horz_iv_ring_in_use[cha], 0xAD, 0x0f, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cms_clockticks[cha], 0xc0, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CHA_LLC_LOOKUP_VICTIMS ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_llc_lookup_rd[cha], 0x34, 0x03, 0x01e20000, 0x10, -1); /* F,M,E,S,I LLC and NM */
      evsetup(fname, &gbl_uncore_perf_fd.fd_llc_lookup_wr[cha], 0x34, 0x05, 0x01e20000, 0x3b, -1); /* F,M,E,S,I LLC and NM */
      evsetup(fname, &gbl_uncore_perf_fd.fd_llc_victims[cha],   0x37, 0x2f, 0x00000000, 0x00, -1); /* F,M,E,S,I LLC and NM */
      /*evsetup(fname, &gbl_uncore_perf_fd.fd_llc_victims[cha],   0x34, 0x11, 0x01e20000, 0x10, -1);*/ /* F,M,E,S,I LLC and NM */
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CHA_XSNP_RESP ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_xsnp_resp[cha], 0x32, 0xff, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CHA_CORE_SNP ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_core_snp[cha], 0x33, 0xe7, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CHA_SNOOPS_SENT ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_snoops_sent[cha], 0x51, 0x01, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CHA_SNOOP_RESP_ALL ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_snoop_resp[cha], 0x5c, 0xff, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_snoop_resp_local[cha], 0x5d, 0xff, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CHA_OSB ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_osb[cha], 0x55, 0x00, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00, 0x00, 0x00, -1);
    } else if ( exp == CTRS_EXP_CHA_TOR ) {
      evsetup(fname, &gbl_uncore_perf_fd.fd_tor_inserts[cha], 0x35, 0x25, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_tor_occupancy[cha], 0x36, 0x24, 0x00, 0x00, -1);
      evsetup(fname, &gbl_uncore_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00, 0x00, 0x00, -1);
    } else {
      /* nothing */
    }
#endif
  }

  gbl_uncore_perf_fd.exp = exp;
}

void setup_core_ctrs( ctrs_core_exp exp ) {
  int ret;
  char fname[1024];
  int core;

#if 0
  snprintf(fname, sizeof(fname), "/sys/devices/cpu");
#else
  sprintf(fname, "/sys/devices/cpu");
#endif

  for ( core = 0; core < CTRS_NCORE; ++core ) {
#ifdef CTRS_CPU_SKX
    if ( exp == CTRS_EXP_L2_BW ) {
      evsetup(fname, &gbl_core_perf_fd.fd_clockticks[core], 0x3c, 0x00, 0x00, 0x00, core);
      evsetup(fname, &gbl_core_perf_fd.fd_l2_lines_in[core], 0xf1, 0x1f, 0x00, 0x00, core);
      evsetup(fname, &gbl_core_perf_fd.fd_l2_lines_in[core], 0xf1, 0x1f, 0x00, 0x00, core);
      evsetup(fname, &gbl_core_perf_fd.fd_l2_lines_out_ns[core], 0xf2, 0x02, 0x00, 0x00, core);
      evsetup(fname, &gbl_core_perf_fd.fd_idi_misc_wb_up[core], 0xfe, 0x02, 0x00, 0x00, core);
      evsetup(fname, &gbl_core_perf_fd.fd_idi_misc_wb_down[core], 0xfe, 0x04, 0x00, 0x00, core);
    } else {
      /* nothing */
    }
#endif
  }

  gbl_core_perf_fd.exp = exp;
}

static uint64_t readctr(int fd) {
  uint64_t data;
  size_t s = read(fd, &data, sizeof(data));

  if (s != sizeof(uint64_t)) {
    err(1, "read counter %lu", s);
  }

  return data;
}

void read_uncore_ctrs( ctrs_uncore *c ) {
  int mc, cha;
  for ( mc = 0; mc < CTRS_NIMC; ++mc ) {
    if ( gbl_uncore_perf_fd.exp == CTRS_EXP_DRAM_ACT ) {
      c->act_rd[mc] = readctr(gbl_uncore_perf_fd.fd_act_rd[mc]);
      c->act_wr[mc] = readctr(gbl_uncore_perf_fd.fd_act_wr[mc]);
      c->imc_clockticks[mc] = readctr(gbl_uncore_perf_fd.fd_imc_clockticks[mc]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_DRAM_CAS ) {
      c->cas_rd[mc] = readctr(gbl_uncore_perf_fd.fd_cas_rd[mc]);
      c->cas_wr[mc] = readctr(gbl_uncore_perf_fd.fd_cas_wr[mc]);
      c->imc_clockticks[mc] = readctr(gbl_uncore_perf_fd.fd_imc_clockticks[mc]);
    } else {
      /* nothing */
    }
  }

  for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
    if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CHA_ACT ) {
      c->cha_rd[cha] = readctr(gbl_uncore_perf_fd.fd_cha_rd[cha]);
      c->cha_wr[cha] = readctr(gbl_uncore_perf_fd.fd_cha_wr[cha]);
      c->cha_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CMS_BL ) {
      c->vert_bl_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_vert_bl_ring_in_use[cha]);
      c->horz_bl_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_horz_bl_ring_in_use[cha]);
      c->cms_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cms_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CMS_AK ) {
      c->vert_ak_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_vert_ak_ring_in_use[cha]);
      c->horz_ak_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_horz_ak_ring_in_use[cha]);
      c->cms_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cms_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CMS_IV ) {
      c->vert_iv_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_vert_iv_ring_in_use[cha]);
      c->horz_iv_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_horz_iv_ring_in_use[cha]);
      c->cms_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cms_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CMS_AK_IV ) {
      c->vert_ak_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_vert_ak_ring_in_use[cha]);
      c->horz_ak_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_horz_ak_ring_in_use[cha]);
      c->vert_iv_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_vert_iv_ring_in_use[cha]);
      c->horz_iv_ring_in_use[cha] = readctr(gbl_uncore_perf_fd.fd_horz_iv_ring_in_use[cha]);
      c->cms_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cms_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CHA_LLC_LOOKUP_VICTIMS ) {
      c->llc_lookup_rd[cha] = readctr(gbl_uncore_perf_fd.fd_llc_lookup_rd[cha]);
      c->llc_lookup_wr[cha] = readctr(gbl_uncore_perf_fd.fd_llc_lookup_wr[cha]);
      c->llc_victims[cha] = readctr(gbl_uncore_perf_fd.fd_llc_victims[cha]);
      c->cha_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CHA_XSNP_RESP ) {
      c->xsnp_resp[cha] = readctr(gbl_uncore_perf_fd.fd_xsnp_resp[cha]);
      c->cha_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CHA_CORE_SNP ) {
      c->core_snp[cha] = readctr(gbl_uncore_perf_fd.fd_core_snp[cha]);
      c->cha_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CHA_SNOOPS_SENT ) {
      c->snoops_sent[cha] = readctr(gbl_uncore_perf_fd.fd_snoops_sent[cha]);
      c->cha_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CHA_SNOOP_RESP_ALL ) {
      c->snoop_resp[cha] = readctr(gbl_uncore_perf_fd.fd_snoop_resp[cha]);
      c->snoop_resp_local[cha] = readctr(gbl_uncore_perf_fd.fd_snoop_resp_local[cha]);
      c->cha_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CHA_OSB ) {
      c->osb[cha] = readctr(gbl_uncore_perf_fd.fd_osb[cha]);
      c->cha_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_uncore_perf_fd.exp == CTRS_EXP_CHA_TOR ) {
      c->tor_inserts[cha] = readctr(gbl_uncore_perf_fd.fd_tor_inserts[cha]);
      c->tor_occupancy[cha] = readctr(gbl_uncore_perf_fd.fd_tor_occupancy[cha]);
      c->cha_clockticks[cha] = readctr(gbl_uncore_perf_fd.fd_cha_clockticks[cha]);
    } else {
      /* nothing */
    }
  }

  c->exp = gbl_uncore_perf_fd.exp;
}

void read_core_ctrs( ctrs_core *c ) {
  int core;
  for ( core = 0; core < CTRS_NCORE; ++core ) {
    if ( gbl_core_perf_fd.exp == CTRS_EXP_L2_BW ) {
      c->clockticks[core] = readctr(gbl_core_perf_fd.fd_clockticks[core]);
#if 1
      c->l2_lines_in[core] = readctr(gbl_core_perf_fd.fd_l2_lines_in[core]);
      c->l2_lines_out_ns[core] = readctr(gbl_core_perf_fd.fd_l2_lines_out_ns[core]);
      c->idi_misc_wb_up[core] = readctr(gbl_core_perf_fd.fd_idi_misc_wb_up[core]);
      c->idi_misc_wb_down[core] = readctr(gbl_core_perf_fd.fd_idi_misc_wb_down[core]);
#else
      c->l2_lines_in[core] = 0;
      c->l2_lines_out_ns[core] = 0;
      c->idi_misc_wb_up[core] = 0;
      c->idi_misc_wb_down[core] = 0;
#endif
    } else {
      /* nothing */
    }
  }

  c->exp = gbl_core_perf_fd.exp;
}

void zero_uncore_ctrs( ctrs_uncore *c ) {
  int mc, cha;
  for ( mc = 0; mc < CTRS_NIMC; ++mc ) {
    c->act_rd[mc] = 0;
    c->act_wr[mc] = 0;
    c->cas_rd[mc] = 0;
    c->cas_wr[mc] = 0;
    c->imc_clockticks[mc] = 0;
  }

  for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
    c->cha_rd[cha] = 0;
    c->cha_wr[cha] = 0;
    c->vert_bl_ring_in_use[cha] = 0;
    c->horz_bl_ring_in_use[cha] = 0;
    c->vert_ak_ring_in_use[cha] = 0;
    c->horz_ak_ring_in_use[cha] = 0;
    c->vert_iv_ring_in_use[cha] = 0;
    c->horz_iv_ring_in_use[cha] = 0;
    c->llc_lookup_rd[cha] = 0;
    c->llc_lookup_wr[cha] = 0;
    c->llc_victims[cha] = 0;
    c->xsnp_resp[cha] = 0;
    c->core_snp[cha] = 0;
    c->snoops_sent[cha] = 0;
    c->snoop_resp[cha] = 0;
    c->snoop_resp_local[cha] = 0;
    c->osb[cha] = 0;
    c->tor_inserts[cha] = 0;
    c->tor_occupancy[cha] = 0;
    c->cha_clockticks[cha] = 0;
    c->cms_clockticks[cha] = 0;
  }
}

void zero_core_ctrs( ctrs_core *c ) {
  int core;
  for ( core = 0; core < CTRS_NCORE; ++core ) {
    c->clockticks[core] = 0;
    c->l2_lines_in[core] = 0;
    c->l2_lines_out_ns[core] = 0;
    c->idi_misc_wb_up[core] = 0;
    c->idi_misc_wb_down[core] = 0;
  }
}

void divi_uncore_ctrs( ctrs_uncore *c, uint64_t div ) {
  int mc, cha;
  for ( mc = 0; mc < CTRS_NIMC; ++mc ) {
    c->act_rd[mc] /= div;
    c->act_wr[mc] /= div;
    c->cas_rd[mc] /= div;
    c->cas_wr[mc] /= div;
    c->imc_clockticks[mc] /= div;
  }

  for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
    c->cha_rd[cha] /= div;
    c->cha_wr[cha] /= div;
    c->vert_bl_ring_in_use[cha] /= div;
    c->horz_bl_ring_in_use[cha] /= div;
    c->vert_ak_ring_in_use[cha] /= div;
    c->horz_ak_ring_in_use[cha] /= div;
    c->vert_iv_ring_in_use[cha] /= div;
    c->horz_iv_ring_in_use[cha] /= div;
    c->llc_lookup_rd[cha] /= div;
    c->llc_lookup_wr[cha] /= div;
    c->llc_victims[cha] /= div;
    c->xsnp_resp[cha] /= div;
    c->core_snp[cha] /= div;
    c->snoops_sent[cha] /= div;
    c->snoop_resp[cha] /= div;
    c->snoop_resp_local[cha] /= div;
    c->osb[cha] /= div;
    c->tor_inserts[cha] /= div;
    c->tor_occupancy[cha] /= div;
    c->cha_clockticks[cha] /= div;
    c->cms_clockticks[cha] /= div;
  }
}

void divi_core_ctrs( ctrs_core *c, uint64_t div ) {
  int core;
  for ( core = 0; core < CTRS_NCORE; ++core ) {
    c->clockticks[core] /= div;
    c->l2_lines_in[core] /= div;
    c->l2_lines_out_ns[core] /= div;
    c->idi_misc_wb_up[core] /= div;
    c->idi_misc_wb_down[core] /= div;
  }
}

void difa_uncore_ctrs( const ctrs_uncore *a, const ctrs_uncore *b, ctrs_uncore* c ) {
  int mc, cha;

  if ( a->exp != b->exp ) {
    printf("exp type for a and b need to be identical!\n");
    return;
  }

  for ( mc = 0; mc < CTRS_NIMC; ++mc ) {
    c->act_rd[mc] += b->act_rd[mc] - a->act_rd[mc];
    c->act_wr[mc] += b->act_wr[mc] - a->act_wr[mc];
    c->cas_rd[mc] += b->cas_rd[mc] - a->cas_rd[mc];
    c->cas_wr[mc] += b->cas_wr[mc] - a->cas_wr[mc];
    c->imc_clockticks[mc] += b->imc_clockticks[mc] - a->imc_clockticks[mc];
  }

  for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
    c->cha_rd[cha] += b->cha_rd[cha] - a->cha_rd[cha];
    c->cha_wr[cha] += b->cha_wr[cha] - a->cha_wr[cha];
    c->vert_bl_ring_in_use[cha] += b->vert_bl_ring_in_use[cha] - a->vert_bl_ring_in_use[cha];
    c->horz_bl_ring_in_use[cha] += b->horz_bl_ring_in_use[cha] - a->horz_bl_ring_in_use[cha];
    c->vert_ak_ring_in_use[cha] += b->vert_ak_ring_in_use[cha] - a->vert_ak_ring_in_use[cha];
    c->horz_ak_ring_in_use[cha] += b->horz_ak_ring_in_use[cha] - a->horz_ak_ring_in_use[cha];
    c->vert_iv_ring_in_use[cha] += b->vert_iv_ring_in_use[cha] - a->vert_iv_ring_in_use[cha];
    c->horz_iv_ring_in_use[cha] += b->horz_iv_ring_in_use[cha] - a->horz_iv_ring_in_use[cha];
    c->llc_lookup_rd[cha] += b->llc_lookup_rd[cha] - a->llc_lookup_rd[cha];
    c->llc_lookup_wr[cha] += b->llc_lookup_wr[cha] - a->llc_lookup_wr[cha];
    c->llc_victims[cha] += b->llc_victims[cha] - a->llc_victims[cha];
    c->xsnp_resp[cha] += b->xsnp_resp[cha] - a->xsnp_resp[cha];
    c->core_snp[cha] += b->core_snp[cha] - a->core_snp[cha];
    c->snoops_sent[cha] += b->snoops_sent[cha] - a->snoops_sent[cha];
    c->snoop_resp[cha] += b->snoop_resp[cha] - a->snoop_resp[cha];
    c->snoop_resp_local[cha] += b->snoop_resp_local[cha] - a->snoop_resp_local[cha];
    c->osb[cha] += b->osb[cha] - a->osb[cha];
    c->tor_inserts[cha] += b->tor_inserts[cha] - a->tor_inserts[cha];
    c->tor_occupancy[cha] += b->tor_occupancy[cha] - a->tor_occupancy[cha];
    c->cha_clockticks[cha] += b->cha_clockticks[cha] - a->cha_clockticks[cha];
    c->cms_clockticks[cha] += b->cms_clockticks[cha] - a->cms_clockticks[cha];
  }

  c->exp = a->exp;
}

void difa_core_ctrs( const ctrs_core *a, const ctrs_core *b, ctrs_core* c ) {
  int core;

  if ( a->exp != b->exp ) {
    printf("exp type for a and b need to be identical!\n");
    return;
  }

  for ( core = 0; core < CTRS_NCORE; ++core ) {
    c->clockticks[core] += b->clockticks[core] - a->clockticks[core];
    c->l2_lines_in[core] += b->l2_lines_in[core] - a->l2_lines_in[core];
    c->l2_lines_out_ns[core] += b->l2_lines_out_ns[core] - a->l2_lines_out_ns[core];
    c->idi_misc_wb_up[core] += b->idi_misc_wb_up[core] - a->idi_misc_wb_up[core];
    c->idi_misc_wb_down[core] += b->idi_misc_wb_down[core] - a->idi_misc_wb_down[core];
  }

  c->exp = a->exp;
}

void get_cas_ddr_bw_uncore_ctrs( const ctrs_uncore *c, const double t, bw_gibs* bw ) {
  uint64_t read_bytes;
  uint64_t write_bytes;
  int mc;

  read_bytes  = 0;
  write_bytes = 0;

  if ( c->exp != CTRS_EXP_DRAM_CAS ) {
    printf("exp type need to be CTRS_EXP_DRAM_CAS!\n");
    bw->rd = 0;
    bw->rd2 = 0;
    bw->wr = 0;
    bw->wr2 = 0;
    return;
  }

  for ( mc = 0; mc < CTRS_NIMC; ++mc ) {
    read_bytes  += c->cas_rd[mc]*64;
    write_bytes += c->cas_wr[mc]*64;
  }

  bw->rd = (((double)read_bytes )/t)/(1024.0*1024.0*1024.0);
  bw->wr = (((double)write_bytes)/t)/(1024.0*1024.0*1024.0);
  bw->wr2 = 0;
}

void get_act_ddr_bw_uncore_ctrs( const ctrs_uncore *c, const double t, bw_gibs* bw ) {
  uint64_t read_bytes;
  uint64_t write_bytes;
  int mc;

  read_bytes  = 0;
  write_bytes = 0;

  if ( c->exp != CTRS_EXP_DRAM_ACT ) {
    printf("exp type need to be CTRS_EXP_DRAM_CAS!\n");
    bw->rd = 0;
    bw->rd2 = 0;
    bw->wr = 0;
    bw->wr2 = 0;
    return;
  }

  for ( mc = 0; mc < CTRS_NIMC; ++mc ) {
    read_bytes  += c->act_rd[mc]*64;
    write_bytes += c->act_wr[mc]*64;
  }

  bw->rd = (((double)read_bytes )/t)/(1024.0*1024.0*1024.0);
  bw->wr = (((double)write_bytes)/t)/(1024.0*1024.0*1024.0);
  bw->wr2 = 0;
}

void get_llc_bw_uncore_ctrs( const ctrs_uncore *c, const double t, bw_gibs* bw ) {
  uint64_t read_bytes;
  uint64_t write_bytes;
  uint64_t victim_bytes;
  int cha;

  read_bytes  = 0;
  write_bytes = 0;
  victim_bytes = 0;

  if ( c->exp != CTRS_EXP_CHA_LLC_LOOKUP_VICTIMS ) {
    printf("exp type need to be CTRS_EXP_CHA_LLC_LOOKUP!\n");
    bw->rd = 0;
    bw->rd2 = 0;
    bw->wr = 0;
    bw->wr2 = 0;
    return;
  }

  for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
    read_bytes  += c->llc_lookup_rd[cha]*64;
    write_bytes += c->llc_lookup_wr[cha]*64;
    victim_bytes += c->llc_victims[cha]*64;
  }

  bw->rd = (((double)read_bytes )/t)/(1024.0*1024.0*1024.0);
  bw->wr = (((double)write_bytes)/t)/(1024.0*1024.0*1024.0);
  bw->wr2 = (((double)victim_bytes)/t)/(1024.0*1024.0*1024.0);
}

void get_l2_bw_core_ctrs( const ctrs_core *c, const double t, bw_gibs* bw ) {
  uint64_t read_bytes;
  uint64_t write_bytes1;
  uint64_t write_bytes2;
  uint64_t write_bytes3;
  int core;

  read_bytes  = 0;
  write_bytes1 = 0;
  write_bytes2 = 0;
  write_bytes3 = 0;

  if ( c->exp != CTRS_EXP_L2_BW ) {
    printf("exp type need to be CTRS_EXP_L2_BW!\n");
    bw->rd = 0;
    bw->rd2 = 0;
    bw->wr = 0;
    bw->wr2 = 0;
    bw->wr3 = 0;
    return;
  }

  for ( core = 0; core < CTRS_NCORE; ++core ) {
    read_bytes   += c->l2_lines_in[core]*64;
    write_bytes1 += c->l2_lines_out_ns[core]*64;
    write_bytes2 += c->idi_misc_wb_up[core]*64;
    write_bytes3 += c->idi_misc_wb_down[core]*64;
  }

  bw->rd = (((double)read_bytes )/t)/(1024.0*1024.0*1024.0);
  bw->rd2 = 0;
  bw->wr = (((double)write_bytes1)/t)/(1024.0*1024.0*1024.0);
  bw->wr2 = (((double)write_bytes2)/t)/(1024.0*1024.0*1024.0);
  bw->wr3 = (((double)write_bytes3)/t)/(1024.0*1024.0*1024.0);
}

void get_l2_bytecycle_core_ctrs( const ctrs_core *c, bw_bc* bw ) {
  uint64_t total_cycles;
  uint64_t read_bytes;
  uint64_t write_bytes1;
  uint64_t write_bytes2;
  uint64_t write_bytes3;
  double   avg_cycles;
  int core;

  total_cycles = 0;
  avg_cycles = 0;
  read_bytes  = 0;
  write_bytes1 = 0;
  write_bytes2 = 0;
  write_bytes3 = 0;

  if ( c->exp != CTRS_EXP_L2_BW ) {
    printf("exp type need to be CTRS_EXP_L2_BW!\n");
    bw->rd = 0;
    bw->rd2 = 0;
    bw->wr = 0;
    bw->wr2 = 0;
    bw->wr3 = 0;
    return;
  }

  for ( core = 0; core < CTRS_NCORE; ++core ) {
    total_cycles += c->clockticks[core];
    read_bytes   += c->l2_lines_in[core]*64;
    write_bytes1 += c->l2_lines_out_ns[core]*64;
    write_bytes2 += c->idi_misc_wb_up[core]*64;
    write_bytes3 += c->idi_misc_wb_down[core]*64;
  }

  avg_cycles = ((double)total_cycles/(double)CTRS_NCORE);

  bw->cyc = avg_cycles;
  bw->rd = ((double)read_bytes/avg_cycles);
  bw->rd2 = 0;
  bw->wr = ((double)write_bytes1/avg_cycles);
  bw->wr2 = ((double)write_bytes2/avg_cycles);
  bw->wr3 = ((double)write_bytes3/avg_cycles);
}

#ifdef __cplusplus
}
#endif

