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


#ifdef USE_LMDB
#include "db_lmdb.hpp"

#include <sys/stat.h>

#include <string>

void LMDB::Open(const string& source) {
  MDB_CHECK(mdb_env_create(&mdb_env_));

  int flags = MDB_RDONLY | MDB_NOTLS;
  flags |= MDB_NOLOCK;
  int rc = mdb_env_open(mdb_env_, source.c_str(), flags, 0664);
  MDB_CHECK(rc);
  printf("Opened lmdb %s\n", source.c_str());
}

LMDBCursor* LMDB::NewCursor() {
  MDB_txn* mdb_txn;
  MDB_stat stat;
  MDB_cursor* mdb_cursor;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
  MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
  MDB_CHECK(mdb_stat(mdb_txn, mdb_dbi_, &stat));
  int count = stat.ms_entries;
  printf("lmdb Database has %d files\n", count);
  return new LMDBCursor(mdb_txn, mdb_cursor, count);
}

#endif  // USE_LMDB
