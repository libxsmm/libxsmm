
// SPDX-License-Identifier: GPL-2.0
/* Copyright(c) 2019 Intel Corporation. All rights reserved. */
#include <stddef.h>
#include <search.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>
#include <accel-config/libaccel_config.h>
#include <cassert>

#include "dsa.h"
#include "utils.h"

dsa_interface::dsa_interface()
{
  descs = 0;
  comp_recs = 0;
  devcnt = 0;
  for (int i = 0; i < MAX_DEVICES; i++) {
    memset(&idxd_wqi_arr[i],0,sizeof(struct idxd_wq_info));
  }
  wqt = NO_WQ;
  lock = PTHREAD_MUTEX_INITIALIZER;
  next_dev = 0;
}

dsa_interface::dsa_interface(long glbsz, long psz, long vecsz)
{
  glb_num_batches = glbsz;
  devcnt = 0;
  wqt = NO_WQ;
  lock = PTHREAD_MUTEX_INITIALIZER;
  next_dev = 0;

  memset(&idxd_wqi_arr[0],0,sizeof(struct idxd_wq_info)*MAX_DEVICES);

  // Allocate memory for the descriptor rings
  // +1 is for the batch descriptor itself
  descs = (struct dsa_hw_desc **)my_malloc(sizeof(struct dsa_hw_desc *)*glb_num_batches, sizeof(void *));
  comp_recs = (struct dsa_completion_record **)my_malloc(sizeof(struct dsa_completion_record *)* \
      glb_num_batches, sizeof(void *));
  for (int i = 0; i < glb_num_batches; i++) {
    descs[i] = (struct dsa_hw_desc *)my_malloc(sizeof(struct dsa_hw_desc)*(2*psz+1),
        sizeof(struct dsa_hw_desc));
    memset(descs[i], 0, sizeof(struct dsa_hw_desc)*(2*psz+1));
    comp_recs[i] = (struct dsa_completion_record *)my_malloc(sizeof(struct \
          dsa_completion_record)*(2*psz+1),
          sizeof(struct dsa_completion_record));
    memset(comp_recs[i], 0, sizeof(struct dsa_completion_record)*(2*psz+1));

    for (int j = 0; j < 2*psz+1; j++) {
      /* CRAV should be 1 so errors can be notified */
      (descs[i])[j].flags = IDXD_OP_FLAG_CRAV;
      /* Request a completion since we poll on status, this flag
         needs to be 1 for status to be updated on successful
         completion */
      (descs[i])[j].flags |= IDXD_OP_FLAG_RCR;
      (descs[i])[j].completion_addr = (unsigned long)&((comp_recs[i])[j]);

      // 1st entry is the batch descriptor itself
      // For all other desc we will compute CRC to mimic "read-only" pattern
      // along with some compute (proxy for reduce)
      // The last descriptor in the batch does a memcopy to model the R/W aspect
      if (j==0) {
        (descs[i])[j].opcode = DSA_OPCODE_BATCH;
        (descs[i])[j].desc_list_addr = (unsigned long)&((descs[i])[1]);
        (descs[i])[j].desc_count = 2*psz;
      } else if (j == 1) {
        (descs[i])[j].opcode = DSA_OPCODE_NOOP;
      } else {
        (descs[i])[j].xfer_size = vecsz;
      }
    }
  }
}

dsa_interface::~dsa_interface()
{
  for (int i = 0; i < glb_num_batches; i++) {
    my_free(descs[i]);
    my_free(comp_recs[i]);
  }
  my_free(descs);
  my_free(comp_recs);
}

#ifndef __NR_getcpu
#define __NR_getcpu     309
#endif

bool dsa_interface::init()
{
  int nodeid;
  syscall(__NR_getcpu, NULL, &nodeid, NULL);


  wqt = DWQ;
  for (int i = 0; i < MAX_DEVICES; i++) {
    char dname[6];
    snprintf(dname, 6, "dsa%d",i*2);
    if (idxd_wq_get(dname, -1, (wqt == SWQ), nodeid))
      devcnt++;
  }
  return (devcnt > 0);
}

void
dsa_interface::wqi_free(void *ptr)
{
  struct idxd_wq_info *wqi;
  size_t n = ARRAY_SIZE(idxd_wqi_arr);

  wqi = (struct idxd_wq_info *)lfind(ptr, idxd_wqi_arr, &n, sizeof(*idxd_wqi_arr), wqi_cmp);
  if (wqi)
    wqi->ptr = NULL;
}

int
dsa_interface::open_wq(struct accfg_wq *wq)
{
  int fd;
  char path[PATH_MAX];
  int rc;

  rc = accfg_wq_get_user_dev_path(wq, path, sizeof(path));
  if (rc)
    return rc;

  fd = open(path, O_RDWR);
  if (fd < 0) {
    ERR("File open error %s: %s\n", path, strerror(errno));
    return -1;
  }

  return fd;
}


struct accfg_wq *
dsa_interface::idxd_wq_find(struct accfg_ctx *ctx, char *dname, int wq_id, int shared, int numa_node)
{
  struct accfg_device *device;
  struct accfg_wq *wq;

  accfg_device_foreach(ctx, device) {
    enum accfg_device_state dstate;
    int fd;

    /* Make sure that the device is enabled */
    dstate = accfg_device_get_state(device);
    if (dstate != ACCFG_DEVICE_ENABLED)
      continue;

    /* Match the device to the id requested */
    if (dname && strcmp(accfg_device_get_devname(device), dname))
      continue;

    if ((!dname && accfg_device_get_numa_node(device) != -1) &&
      (numa_node != accfg_device_get_numa_node(device)))
      continue;

    accfg_wq_foreach(device, wq) {
      enum accfg_wq_state wstate;
      enum accfg_wq_type type;

      if (wq_id != -1 && accfg_wq_get_id(wq) != wq_id)
        continue;

      /* Get a workqueue that's enabled */
      wstate = accfg_wq_get_state(wq);
      if (wstate != ACCFG_WQ_ENABLED)
        continue;

      /* The wq type should be user */
      type = accfg_wq_get_type(wq);
      if (type != ACCFG_WQT_USER)
        continue;

      /* Make sure the mode is correct */
      if (wq_id == -1) {
        int mode = accfg_wq_get_mode(wq);

        if ((mode == ACCFG_WQ_SHARED && !shared)
          || (mode == ACCFG_WQ_DEDICATED && shared))
          continue;
      }

      fd = open_wq(wq);
      if (fd < 0)
        continue;

      close(fd);
      return wq;
    }
  }

  return NULL;
}

void *
dsa_interface::idxd_wq_mmap(struct accfg_wq *wq)
{
  int fd;
  void *wq_reg;

  fd = open_wq(wq);

  wq_reg = mmap(NULL, 0x1000, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
  if (wq_reg == MAP_FAILED) {
    ERR("mmap error: %s", strerror(errno));
    close(fd);
    return NULL;
  }

  close(fd);
  return wq_reg;
}

void *
dsa_interface::idxd_wq_get(char *dname, int wq_id, int shared, int numa_node)
{
  struct accfg_ctx *ctx;
  struct accfg_wq *wq;
  struct idxd_wq_info *wqi;
  void *ptr;

  pthread_mutex_lock(&lock);

  accfg_new(&ctx);

  wqi = wqi_alloc();

  wq = idxd_wq_find(ctx, dname, wq_id, shared, numa_node);
  if (wq == NULL) {
    //ERR("Failed to find a WQ\n");
    goto err_ret;
  }

  ptr = idxd_wq_mmap(wq);
  if (!ptr) {
    ERR("Failed to map WQ dev %s wq %d\n", dname,
      accfg_wq_get_id(wq));
    goto err_ret;
  }

  pthread_mutex_unlock(&lock);

  wqi->wq = wq;
  wqi->ptr = ptr;
  wqi->ctx = ctx;

  return ptr;

 err_ret:
  pthread_mutex_unlock(&lock);
  accfg_unref(ctx);
  return NULL;
}

void
dsa_interface::idxd_wq_info_get(void *ptr, struct wq_info *wq_info)
{
  struct idxd_wq_info *wqi;
  size_t n = ARRAY_SIZE(idxd_wqi_arr);

  wqi = (struct idxd_wq_info *)lfind(ptr, idxd_wqi_arr, &n, sizeof(*idxd_wqi_arr), wqi_cmp);

  wq_info->size = accfg_wq_get_size((struct accfg_wq *)(wqi->wq));
  wq_info->dmap_fd = -1;
  wq_info->dname = accfg_device_get_devname(accfg_wq_get_device((struct accfg_wq *)wqi->wq));
  wq_info->dwq = accfg_wq_get_mode((struct accfg_wq *)wqi->wq) == ACCFG_WQ_DEDICATED;
  wq_info->dev_type = 0;
}

void
dsa_interface::idxd_wq_unmap(void *wq)
{
  struct idxd_wq_info *wqi;
  size_t n = ARRAY_SIZE(idxd_wqi_arr);

  pthread_mutex_lock(&lock);

  wqi = (struct idxd_wq_info *)lfind(wq, idxd_wqi_arr, &n, sizeof(*idxd_wqi_arr), wqi_cmp);
  wqi_free(wqi);
  munmap(wq, 0x1000);
  accfg_unref((struct accfg_ctx *)(wqi->ctx));

  pthread_mutex_unlock(&lock);
}

void dsa_interface::prep_gather(int bid, const long *offsets, const long *indices, unsigned char *src,
         unsigned char *dst, long NS, int vecsz)
{
  int n = bid;
  struct dsa_hw_desc* ring = descs[n];

  auto start = offsets[n];
  auto end = (n < glb_num_batches - 1) ? offsets[n + 1] : NS;

  int i = 2;
  for (long s = start; s < end; s++, i++) {
    ring[i].opcode = DSA_OPCODE_CRCGEN;
    ring[i].src_addr = (unsigned long)&src[indices[s]*vecsz];
    ring[i].dst_addr = 0;
    ring[i].flags &= ~IDXD_OP_FLAG_CC;
    //fprintf(stderr, "src[%d]=0x%8lx\n", i, ring[i].src_addr);
  }
  ring[0].desc_count = i-1;

  ring[i-1].opcode = DSA_OPCODE_MEMMOVE;
  dst[n*vecsz] = 0;
  ring[i-1].dst_addr = (unsigned long)&dst[n*vecsz];
  /* Hint to direct data writes to CPU cache */
  ring[i-1].flags |= IDXD_OP_FLAG_CC;
  ring[i-1].flags |= IDXD_OP_FLAG_BOF;
  //fprintf(stderr, "dst[%d]=0x%8lx\n", i-1, ring[i-1].dst_addr);
}
