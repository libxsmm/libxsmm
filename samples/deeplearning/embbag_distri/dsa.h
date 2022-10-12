// SPDX-License-Identifier: GPL-2.0
/* Copyright(c) 2019 Intel Corporation. All rights reserved. */
#ifndef __DSA_H__
#define __DSA_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <linux/idxd.h>
#include <search.h>

#define ARRAY_SIZE(x) (sizeof((x))/sizeof((x)[0]))
#define ERR(arg...) fprintf(stderr, ## arg)

class dsa_interface
{
  private:
#define EMB_MAX_THREADS 112
#define EMB_MAX_BATCHES_PER_THREAD 16
#define EMB_MAX_POOL 100
#define MAX_DEVICES 8
#define MAX_COMP_RETRY 2000000000

  enum wq_type {
    NO_WQ = 0,
    DWQ = 1,
    SWQ = 2
  };

  struct dsa_hw_desc **descs;
  struct dsa_completion_record **comp_recs;

  struct wq_info {
    const char *dname;
    int dmap_fd;
    int size;
    int dwq;
    int dev_type;
  };

  struct idxd_wq_info {
    void *ptr;
    void *ctx;
    void *wq;
  };

  struct idxd_wq_info idxd_wqi_arr[MAX_DEVICES];
  long glb_num_batches;
  long pool_size;
  int devcnt;
  enum wq_type wqt;
  pthread_mutex_t lock;
  int next_dev;


  inline unsigned char enqcmd(struct dsa_hw_desc *desc,
        volatile void *reg);
  inline void movdir64b(struct dsa_hw_desc *desc, volatile void *reg);
  friend inline int wqi_cmp(const void *p1, const void *p2);
  inline struct idxd_wq_info * wqi_alloc(void);
  void wqi_free(void *ptr);
  int open_wq(struct accfg_wq *wq);
  struct accfg_wq * idxd_wq_find(struct accfg_ctx *ctx, char *dname, int wq_id,
          int shared, int numa_node);
  void * idxd_wq_mmap(struct accfg_wq *wq);


  public:
  dsa_interface();
  dsa_interface(long, long, long);
  ~dsa_interface();
  bool init();
  void prep_gather(int bid, const long *offsets, const long *indices, unsigned char *src,
       unsigned char *dst, long NS, int vecsz);
  bool desc_submit(int bid);
  inline int poll_comp(int, int*);
  inline void dump_desc(struct dsa_hw_desc *hw);
  void * idxd_wq_get(char *dname, int wq_id, int shared, int numa_node);
  void idxd_wq_info_get(void *ptr, struct wq_info *wq_info);
  void idxd_wq_unmap(void *wq);
  inline int get_num_devices();

};

inline int dsa_interface::get_num_devices()
{
  return devcnt;
}

/* Dump DSA hardware descriptor to log */
inline void dsa_interface::dump_desc(struct dsa_hw_desc *hw)
{
  struct dsa_raw_desc *rhw = (struct dsa_raw_desc *)hw;
  int i;

  printf("desc addr: %p\n", hw);

  for (i = 0; i < 8; i++)
    printf("desc[%d]: 0x%016lx\n", i, rhw->field[i]);
}

inline unsigned char dsa_interface::enqcmd(struct dsa_hw_desc *desc,
      volatile void *reg)
{
  unsigned char retry;

  asm volatile(".byte 0xf2, 0x0f, 0x38, 0xf8, 0x02\t\n"
      "setz %0\t\n"
      : "=r"(retry) : "a" (reg), "d" (desc));
  return retry;
}

inline void dsa_interface::movdir64b(struct dsa_hw_desc *desc, volatile void *reg)
{
  asm volatile(".byte 0x66, 0x0f, 0x38, 0xf8, 0x02\t\n"
    : : "a" (reg), "d" (desc));
}

__always_inline
bool dsa_interface::desc_submit(int bid)
{
  assert(wqt != NO_WQ);
  bool ret = true;
  const int max_retry = 10000;
  int retry = 0;

  struct dsa_hw_desc *batch = descs[bid];
  struct dsa_hw_desc *hw = &(batch[0]);

  /* reset completion status */
  comp_recs[bid][0].status = 0;
  __builtin_ia32_sfence();

  void *wq_portal = idxd_wqi_arr[next_dev].ptr;
  if (wqt == DWQ)
    movdir64b(hw, wq_portal);
  else {
    /* retry upto max val */
    while (retry++ < max_retry && !(ret = !enqcmd(hw, wq_portal)))
      __builtin_ia32_pause();
  }
  /* Use the next cache line for the next submission */
  unsigned long base = (unsigned long)(idxd_wqi_arr[next_dev].ptr) & ~(unsigned long)0xFFF;
  unsigned long off = ((unsigned long)(idxd_wqi_arr[next_dev].ptr) + 64) & (unsigned long)0xFFF;
  idxd_wqi_arr[next_dev].ptr = (void *)(base+off);
  next_dev = (next_dev + 1)%devcnt;
  return ret;
}

inline int
wqi_cmp(const void *p1, const void *p2)
{
  const struct dsa_interface::idxd_wq_info *wqi = (const struct dsa_interface::idxd_wq_info *)p2;

  return (void *)(wqi->ptr) == p1 ? 0 : 1;
}

inline struct dsa_interface::idxd_wq_info *
dsa_interface::wqi_alloc(void)
{
  size_t n = ARRAY_SIZE(idxd_wqi_arr);

  return (struct idxd_wq_info *)lfind(NULL, (void *)idxd_wqi_arr, &n, sizeof(*idxd_wqi_arr), wqi_cmp);
}

inline int dsa_interface::poll_comp(int bid, int *ret)
{
  struct dsa_completion_record *comp = &(comp_recs[bid][0]);
  int retry = 0;
  while (comp->status == 0 && retry++ < MAX_COMP_RETRY)
    __builtin_ia32_pause();

  *ret = comp->status;
  if (*ret != DSA_COMP_SUCCESS) {
    ERR("batch%d did not complete\n",bid);
    for (int i = 0; i < descs[bid][0].desc_count; i++) {
      if (comp_recs[bid][i+1].status != DSA_COMP_SUCCESS) {
        ERR("desc%d of %d in batch%d had status=0x%x\n",
          i, descs[bid][0].desc_count, bid, comp_recs[bid][i+1].status);
        dump_desc(&descs[bid][i+1]);
      }
    }
  }
  return !(comp->status == DSA_COMP_SUCCESS);
}
#endif
