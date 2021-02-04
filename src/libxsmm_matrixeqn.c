/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_matrixeqn.h"

/* aux struct for matrix equations */
LIBXSMM_APIVAR_DEFINE(libxsmm_matrix_eqn* libxsmm_matrix_eqns[256]);
LIBXSMM_APIVAR_DEFINE(libxsmm_blasint libxsmm_matrix_eqns_init);
LIBXSMM_APIVAR_DEFINE(libxsmm_blasint libxsmm_matrix_eqns_count);

LIBXSMM_API_INTERN libxsmm_matrix_eqn* libxsmm_matrix_eqn_get_equation( libxsmm_blasint eqn_idx ) {
  return libxsmm_matrix_eqns[eqn_idx];
}

LIBXSMM_API_INTERN void libxsmm_matrix_eqn_assign_reg_scores( libxsmm_matrix_eqn_elem* cur_node );
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_assign_reg_scores( libxsmm_matrix_eqn_elem* cur_node ) {
  /* check if we are at an argument leaf, then we assign register score 0 */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if ( (cur_node->le == NULL) && (cur_node->ri == NULL) ) {
      cur_node->reg_score = 0;
      cur_node->max_tmp_size = cur_node->info.arg.ld * cur_node->info.arg.n;
    }
    else {
      printf("ERROR: Arg cannot have left or right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* If the node is unary type we have two cases:
 *  *      * 1) If the left child is an arg, we just set the score to 1 (we do not overwrite the input)
 *   *           * 2) if the left child is NOT an arg, we just propagate the register score from it (no additional tmp storage is needed) */
    if ( cur_node->le != NULL ) {
      libxsmm_matrix_eqn_assign_reg_scores( cur_node->le );
      cur_node->max_tmp_size = cur_node->le->max_tmp_size;
      if ( cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
        cur_node->reg_score = 1;
      } else {
        cur_node->reg_score = cur_node->le->reg_score;
      }
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( (cur_node->ri != NULL) ) {
      printf("ERROR: Unary cannot have right childs!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* If the node is binary type we have two cases:
 *  *      * 1) If the left/right subtrees have the same register score, we have to increase it by one (i.e. we have to first compute one of the subtrees and keep the result in a tmp storage and then compute the other subtree, so we would need an extra tmp storage)
 *   *           * 2) If the left/right subtrees DO NOT have the same register score, then we assign  the maximum of the register scores (i.e. we would compute first the subtree with the maximum score and then the tree with the smallest score, thus no extra tmp storage is required) */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) ) {
      libxsmm_matrix_eqn_assign_reg_scores( cur_node->le );
      libxsmm_matrix_eqn_assign_reg_scores( cur_node->ri );
      cur_node->max_tmp_size = LIBXSMM_MAX(cur_node->le->max_tmp_size, cur_node->ri->max_tmp_size);
      if (cur_node->le->reg_score == cur_node->ri->reg_score) {
        cur_node->reg_score = cur_node->le->reg_score + 1;
      } else {
        cur_node->reg_score = LIBXSMM_MAX(cur_node->le->reg_score, cur_node->ri->reg_score);
      }
    } else {
      printf("ERROR: Binary needs left and right child!\n");
    }
  } else {
    /* shouldn't happen */
  }
}


LIBXSMM_API_INTERN libxsmm_blasint reserve_tmp_storage(libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool);
LIBXSMM_API_INTERN libxsmm_blasint reserve_tmp_storage(libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool) {
  libxsmm_blasint i;
  if ( tmp_storage_pool != NULL ) {
    for (i = 0; i < n_max_tmp; i++) {
      if (tmp_storage_pool[i] == 0) {
        tmp_storage_pool[i] = 1;
        return i;
      }
    }
  }
  return -1;
}


LIBXSMM_API_INTERN void libxsmm_matrix_eqn_create_exec_plan( libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool );
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_create_exec_plan( libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool ) {
  /* check if we are at an argument leaf, then we assign register score 0 */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    /* Do not increase the timestamp, this node is just an arg so it's not part of the execution */
    cur_node->visit_timestamp = -1;
    cur_node->n_args = 1;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_matrix_eqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
    cur_node->visit_timestamp = *global_timestamp;
    *global_timestamp = *global_timestamp + 1;
    cur_node->n_args = cur_node->le->n_args;
    /* When assigning the tmp output storage, we have two cases in the unary:
 *  *      * 1) The child is an arg, so we have to reserve a tmp storage
 *   *           * 2) The child is NOT an arg, so we just reuse the tmp storage of the child */
    if ( cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
      cur_node->le->up = cur_node;
      cur_node->tmp.id = reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
      cur_node->tmp.m  = cur_node->le->info.arg.m;
      cur_node->tmp.n  = cur_node->le->info.arg.n;
      cur_node->tmp.ld  = cur_node->le->info.arg.ld;
      cur_node->tmp.dtype  = cur_node->le->info.arg.dtype;
      cur_node->tree_max_comp_tsize = LIBXSMM_TYPESIZE( cur_node->info.u_op.dtype );
    } else {
      cur_node->tmp.id = cur_node->le->tmp.id;
      cur_node->tmp.m  = cur_node->le->tmp.m;
      cur_node->tmp.n  = cur_node->le->tmp.n;
      cur_node->tmp.ld  = cur_node->le->tmp.ld;
      cur_node->tmp.dtype  = cur_node->le->tmp.dtype;
      cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE(cur_node->info.u_op.dtype), cur_node->le->tree_max_comp_tsize );
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* First we visit the child tree with the maximum register score */
    if (cur_node->le->reg_score >= cur_node->ri->reg_score) {
      libxsmm_matrix_eqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
      libxsmm_matrix_eqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
    } else {
      libxsmm_matrix_eqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
      libxsmm_matrix_eqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
    }
    cur_node->visit_timestamp = *global_timestamp;
    *global_timestamp = *global_timestamp + 1;
    cur_node->n_args = cur_node->le->n_args + cur_node->ri->n_args;
    /* When assigning the tmp output storage, we have three cases in the binary:
 *  *      * 1) Both children are arg, so we have to reserve a tmp storage
 *   *           * 2) Both child are NOT arg, so we reuse the tmp storage of either one for our output and we make the other tmp storage available
 *    *                * 3) One child IS arg and the other child is NOT an arg, so we just reuse the tmp storage of the non-arg child */
    if ( (cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
      cur_node->le->up = cur_node;
      cur_node->ri->up = cur_node;
      cur_node->tmp.id = reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
      cur_node->tmp.m  = cur_node->le->info.arg.m;
      cur_node->tmp.n  = cur_node->le->info.arg.n;
      cur_node->tmp.ld  = cur_node->le->info.arg.ld;
      cur_node->tmp.dtype  = cur_node->le->info.arg.dtype;
      cur_node->tree_max_comp_tsize = LIBXSMM_TYPESIZE( cur_node->info.b_op.dtype );
    } else if ( (cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
      cur_node->tmp.id = cur_node->le->tmp.id;
      cur_node->tmp.m  = cur_node->le->tmp.m;
      cur_node->tmp.n  = cur_node->le->tmp.n;
      cur_node->tmp.ld  = cur_node->le->tmp.ld;
      cur_node->tmp.dtype  = cur_node->le->tmp.dtype;
      tmp_storage_pool[cur_node->ri->tmp.id] = 0;
      cur_node->tree_max_comp_tsize = LIBXSMM_MAX( cur_node->ri->tree_max_comp_tsize, cur_node->le->tree_max_comp_tsize );
    } else {
      if (cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) {
        cur_node->ri->up = cur_node;
        cur_node->tmp.id = cur_node->le->tmp.id;
        cur_node->tmp.m  = cur_node->le->tmp.m;
        cur_node->tmp.n  = cur_node->le->tmp.n;
        cur_node->tmp.ld  = cur_node->le->tmp.ld;
        cur_node->tmp.dtype  = cur_node->le->tmp.dtype;
        cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE(cur_node->info.b_op.dtype), cur_node->le->tree_max_comp_tsize );
      } else {
        cur_node->le->up = cur_node;
        cur_node->tmp.id = cur_node->ri->tmp.id;
        cur_node->tmp.m  = cur_node->ri->tmp.m;
        cur_node->tmp.n  = cur_node->ri->tmp.n;
        cur_node->tmp.ld  = cur_node->ri->tmp.ld;
        cur_node->tmp.dtype  = cur_node->ri->tmp.dtype;
        cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE(cur_node->info.b_op.dtype), cur_node->ri->tree_max_comp_tsize );
      }
    }
  } else {
    /* shouldn't happen */
  }
}


LIBXSMM_API_INTERN void libxsmm_matrix_eqn_opt_exec_plan( libxsmm_blasint idx );
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_opt_exec_plan( libxsmm_blasint idx ) {
  libxsmm_blasint global_timestamp = 0;
  libxsmm_blasint max_reg_score = 0;
  libxsmm_blasint *tmp_storage_pool = NULL;
  libxsmm_blasint i;
  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation doesn't exist, nothing to optimize!\n" );
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 0 ) {
    fprintf( stderr, "the requested equation is not yet finalized, so can't optimize!\n" );
  }
#if 0
  printf("\n");
  printf("Assigning register scores to find optimal traversal plan (i.e. that minimizes tmp storage)... \n");
#endif
  libxsmm_matrix_eqn_assign_reg_scores( libxsmm_matrix_eqns[idx]->eqn_root );
  max_reg_score = libxsmm_matrix_eqns[idx]->eqn_root->reg_score;
  tmp_storage_pool = (libxsmm_blasint*) malloc(max_reg_score * sizeof(libxsmm_blasint));
  if (tmp_storage_pool == NULL) {
    fprintf( stderr, "Tmp storage allocation array failed...\n" );
    return;
  } else {
    for (i = 0; i < max_reg_score; i++) {
      tmp_storage_pool[i] = 0;
    }
  }
#if 0
  printf("Optimal number of intermediate tmp storage is %d\n", max_reg_score);
#endif
  libxsmm_matrix_eqn_create_exec_plan( libxsmm_matrix_eqns[idx]->eqn_root, &global_timestamp, max_reg_score, tmp_storage_pool );
#if 0
  printf("Created optimal exexution plan...\n");
#endif
  if (tmp_storage_pool != NULL) {
    free(tmp_storage_pool);
  }
#if 0
  printf("\n\n");
#endif
  libxsmm_matrix_eqns[idx]->is_optimized = 1;
}


LIBXSMM_API_INTERN libxsmm_matrix_eqn_elem* libxsmm_matrix_eqn_add_node( libxsmm_matrix_eqn_elem* cur_node, libxsmm_matrix_eqn_node_type type, libxsmm_matrix_eqn_info info );
LIBXSMM_API_INTERN libxsmm_matrix_eqn_elem* libxsmm_matrix_eqn_add_node( libxsmm_matrix_eqn_elem* cur_node, libxsmm_matrix_eqn_node_type type, libxsmm_matrix_eqn_info info ) {
  if ( type == LIBXSMM_MATRIX_EQN_NODE_NONE ) {
    /* shouldn't happen */
    fprintf( stderr, "wrong op node type to add!\n");
  }

  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_matrix_eqn_elem *node = (libxsmm_matrix_eqn_elem*) malloc( sizeof(libxsmm_matrix_eqn_elem) );

    node->le = NULL;
    node->ri = NULL;
    node->up = cur_node;
    node->type = type;
    node->info = info;

    if ( cur_node->le == NULL ) {
      cur_node->le = node;
    } else {
      /* shouldn't happen */
      fprintf( stderr, "this is not a leaf node, so we cannot add a node!\n");
      free( node );
      node = NULL;
    }

    return node;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    libxsmm_matrix_eqn_elem *node = (libxsmm_matrix_eqn_elem*) malloc( sizeof(libxsmm_matrix_eqn_elem) );

    node->le = NULL;
    node->ri = NULL;
    node->up = cur_node;
    node->type = type;
    node->info = info;

    if ( cur_node->le == NULL ) {
      cur_node->le = node;
    } else if ( cur_node->ri == NULL ) {
      cur_node->ri = node;
    } else {
      /* shouldn't happen */
      fprintf( stderr, "this is not a leaf node, so we cannot add a node!\n");
      free( node );
      node = NULL;
    }

    return node;
  /* we converting the root */
  } else if ( (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_NONE) && (type != LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    cur_node->le = NULL;
    cur_node->ri = NULL;
    cur_node->up = NULL;
    cur_node->type = type;
    cur_node->info = info;

    return cur_node;
  } else {
    /* shouldn't happen */
    fprintf( stderr, "at this position we cannot add an op!\n");
  }

  return NULL;
}


LIBXSMM_API_INTERN libxsmm_matrix_eqn_elem* libxsmm_matrix_eqn_trv_head( libxsmm_matrix_eqn_elem* cur_node );
LIBXSMM_API_INTERN libxsmm_matrix_eqn_elem* libxsmm_matrix_eqn_trv_head( libxsmm_matrix_eqn_elem* cur_node ) {
  /* check if we are at an argument leaf, then we move up */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    return libxsmm_matrix_eqn_trv_head( cur_node->up );
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le == NULL ) {
      return cur_node;
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( cur_node->up == NULL ) {
      return cur_node;
    /* we have to find another node */
    } else {
      return libxsmm_matrix_eqn_trv_head( cur_node->up );
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le == NULL ) {
      return cur_node;
    } else if ( cur_node->ri == NULL ) {
      return cur_node;
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( cur_node->up == NULL ) {
      return cur_node;
    /* we have to find another node */
    } else {
      return libxsmm_matrix_eqn_trv_head( cur_node->up );
    }
  } else {
    /* should not happen */
  }

  return NULL;
}


LIBXSMM_API_INTERN void libxsmm_matrix_eqn_trv_print( libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint indent );
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_trv_print( libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint indent ) {
  libxsmm_blasint i;
  libxsmm_blasint tree_print_indent = 4;

  for ( i = 0; i < indent; ++i ) {
    if ( i < indent - tree_print_indent ) {
      printf(" ");
    } else {
      if ( i % tree_print_indent == 0 ) {
        printf("|");
      } else {
        printf("-");
      }
    }
  }

  /* check if we are at an argument leaf, then we move up */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if ( (cur_node->le == NULL) && (cur_node->ri == NULL) ) {
      printf("ARG: %i %i %i %i %i\n", cur_node->info.arg.m, cur_node->info.arg.n, cur_node->info.arg.ld, cur_node->info.arg.in_pos, cur_node->info.arg.offs_in_pos );
    } else {
      printf("ERROR: Arg cannot have left or right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le != NULL ) {
      printf("UNARY: %i %i (timestamp = %i, tmp = %i)\n", (int)cur_node->info.u_op.type, (int)cur_node->info.u_op.flags, cur_node->visit_timestamp, cur_node->tmp.id );
      libxsmm_matrix_eqn_trv_print( cur_node->le, indent+tree_print_indent );
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( (cur_node->ri != NULL) ) {
      printf("ERROR: Unary cannot have right childs!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* we have to push more in this branch */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) ) {
      printf("BINARY: %i %i (timestamp = %i, tmp = %i)\n", (int)cur_node->info.b_op.type, (int)cur_node->info.b_op.flags, cur_node->visit_timestamp, cur_node->tmp.id );
      libxsmm_matrix_eqn_trv_print( cur_node->le, indent+tree_print_indent );
      libxsmm_matrix_eqn_trv_print( cur_node->ri, indent+tree_print_indent );
    } else {
      printf("ERROR: Binary needs left and right child!\n");
    }
  } else {
    /* shouldn't happen */
  }
}


LIBXSMM_API_INTERN void libxsmm_matrix_eqn_trv_rpn_print( libxsmm_matrix_eqn_elem* cur_node );
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_trv_rpn_print( libxsmm_matrix_eqn_elem* cur_node ) {
  /* check if we are at an argument leaf, then we move up */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if ( (cur_node->le == NULL) && (cur_node->ri == NULL) ) {
      printf("ARG ");
    } else {
      printf("ERROR: Arg cannot have left or right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le != NULL ) {
      libxsmm_matrix_eqn_trv_rpn_print( cur_node->le );
      printf("UNARY-%i ", (int)cur_node->info.u_op.type );
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( (cur_node->ri != NULL) ) {
      printf("ERROR: Unary cannot have right childs!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* we have to push more in this branch */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) ) {
      libxsmm_matrix_eqn_trv_rpn_print( cur_node->le );
      libxsmm_matrix_eqn_trv_rpn_print( cur_node->ri );
      printf("BINARY-%i ", (int)cur_node->info.b_op.type );
    } else {
      printf("ERROR: Binary needs left and right child!\n");
    }
  } else {
    /* shouldn't happen */
  }
}


LIBXSMM_API_INTERN void libxsmm_matrix_eqn_mov_head( libxsmm_blasint idx );
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_mov_head( libxsmm_blasint idx ) {
  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation doesn't exist!\n" );
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
  }

  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_matrix_eqn_trv_head( libxsmm_matrix_eqns[idx]->eqn_cur );

#if 0
  printf("cur node address: %lld\n", libxsmm_matrix_eqns[idx]->eqn_cur );
#endif

  /* let's see if we need seal the equation */
  if ( (libxsmm_matrix_eqns[idx]->eqn_cur == libxsmm_matrix_eqns[idx]->eqn_root) &&
       ( ((libxsmm_matrix_eqns[idx]->eqn_cur->type == LIBXSMM_MATRIX_EQN_NODE_UNARY)  && (libxsmm_matrix_eqns[idx]->eqn_cur->le != NULL)) ||
         ((libxsmm_matrix_eqns[idx]->eqn_cur->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (libxsmm_matrix_eqns[idx]->eqn_cur->ri != NULL))    ) ) {
    libxsmm_matrix_eqns[idx]->is_constructed = 1;
    libxsmm_matrix_eqn_opt_exec_plan( idx );
  }
}


LIBXSMM_API_INTERN int libxsmm_matrix_eqn_is_ready_for_jit( libxsmm_blasint eqn_idx ) {
  if ( libxsmm_matrix_eqns[eqn_idx] == NULL ) {
    fprintf( stderr, "the requested equation doesn't exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[eqn_idx]->is_constructed == 0 ) {
    fprintf( stderr, "the requested equation is not finalized, yet!\n" );
    return 2;
  }
  if ( libxsmm_matrix_eqns[eqn_idx]->is_optimized == 0 ) {
    fprintf( stderr, "the requested equation is not optimized, yet!\n" );
    return 2;
  }

  return 0;
}


LIBXSMM_API libxsmm_blasint libxsmm_matrix_eqn_create() {
  libxsmm_blasint ret = libxsmm_matrix_eqns_count;
  libxsmm_matrix_eqn_elem* node;

  /* lazy init of helper array */
  if ( libxsmm_matrix_eqns_init == 0 ) {
    libxsmm_blasint i;
    for ( i = 0; i < 256; ++i ) {
      libxsmm_matrix_eqns[i] = NULL;
    }
    libxsmm_matrix_eqns_count = 0;
    libxsmm_matrix_eqns_init = 1;
  }

  libxsmm_matrix_eqns_count++;

  libxsmm_matrix_eqns[ret] = (libxsmm_matrix_eqn*) malloc( sizeof(libxsmm_matrix_eqn) );

  node = (libxsmm_matrix_eqn_elem*) malloc( sizeof(libxsmm_matrix_eqn_elem) );

  node->le = NULL;
  node->ri = NULL;
  node->up = NULL;
  node->type = LIBXSMM_MATRIX_EQN_NODE_NONE;

  libxsmm_matrix_eqns[ret]->eqn_root = node;
  libxsmm_matrix_eqns[ret]->eqn_cur = node;
  libxsmm_matrix_eqns[ret]->is_constructed = 0;
  libxsmm_matrix_eqns[ret]->is_optimized = 0;
  libxsmm_matrix_eqns[ret]->unary_only = 0;
  libxsmm_matrix_eqns[ret]->unary_only = 0;
#if 0
  printf("created equation no: %i\n", ret);
  printf("root node address: %lld\n", libxsmm_matrix_eqns[ret]->eqn_cur );
#endif

  return ret;
}


LIBXSMM_API int libxsmm_matrix_eqn_push_back_arg( const libxsmm_blasint idx, const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint ld, const libxsmm_blasint in_pos, const libxsmm_blasint offs_in_pos, const libxsmm_datatype dtype ) {
  union libxsmm_matrix_eqn_info info;

  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation doesn't exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
    return 2;
  }

  info.arg.m = m;
  info.arg.n = n;
  info.arg.ld = ld;
  info.arg.in_pos = in_pos;
  info.arg.offs_in_pos = offs_in_pos;
  info.arg.dtype = dtype;
  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_matrix_eqn_add_node( libxsmm_matrix_eqns[idx]->eqn_cur, LIBXSMM_MATRIX_EQN_NODE_ARG, info );
#if 0
  printf("added arg node: %lld %i %i %i %i %i %i\n", libxsmm_matrix_eqns[idx]->eqn_cur, M, N, ld, in_pos, offs_in_pos, dtype );
#endif

  /* move to the next head position in the tree */
  libxsmm_matrix_eqn_mov_head( idx );

  return 0;
}


LIBXSMM_API int libxsmm_matrix_eqn_push_back_unary_op( const libxsmm_blasint idx, const libxsmm_meltw_unary_type type, const libxsmm_meltw_unary_flags flags, const libxsmm_datatype dtype ) {
  union libxsmm_matrix_eqn_info info;

  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation doesn't exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
    return 2;
  }

  info.u_op.type  = type;
  info.u_op.flags = flags;
  info.u_op.dtype = dtype;
  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_matrix_eqn_add_node( libxsmm_matrix_eqns[idx]->eqn_cur, LIBXSMM_MATRIX_EQN_NODE_UNARY, info );
#if 0
  printf("added unary node: %lld %i %i %i\n", libxsmm_matrix_eqns[idx]->eqn_cur, type, flags, dtype );
#endif

  /* move to the next head position in the tree */
  libxsmm_matrix_eqn_mov_head( idx );

  return 0;
}


LIBXSMM_API int libxsmm_matrix_eqn_push_back_binary_op( const libxsmm_blasint idx, const libxsmm_meltw_binary_type type, const libxsmm_meltw_binary_flags flags, const libxsmm_datatype dtype ) {
  union libxsmm_matrix_eqn_info info;

  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation doesn't exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
    return 2;
  }

  info.b_op.type  = type;
  info.b_op.flags = flags;
  info.b_op.dtype = dtype;
  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_matrix_eqn_add_node( libxsmm_matrix_eqns[idx]->eqn_cur, LIBXSMM_MATRIX_EQN_NODE_BINARY, info );
#if 0
  printf("added binary node: %lld %i %i %i\n", libxsmm_matrix_eqns[idx]->eqn_cur, type, flags, dtype );
#endif

  /* move to the next head position in the tree */
  libxsmm_matrix_eqn_mov_head( idx );

  return 0;
}


LIBXSMM_API void libxsmm_matrix_eqn_tree_print( const libxsmm_blasint idx ) {
  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation doesn't exist!\n" );
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 0 ) {
    fprintf( stderr, "the requested equation is not yet finalized!\n" );
  }

  printf("\n");
  printf("Schematic of the expression tree (Pre-order)\n");
  libxsmm_matrix_eqn_trv_print( libxsmm_matrix_eqns[idx]->eqn_root, 0 );
  printf("\n");
}


LIBXSMM_API void libxsmm_matrix_eqn_rpn_print( const libxsmm_blasint idx ) {
  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation doesn't exist!\n" );
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 0 ) {
    fprintf( stderr, "the requested equation is not yet finalized!\n" );
  }

  printf("\n");
  printf("HP calculator (RPN) print of the expression tree (Post-order)\n");
  libxsmm_matrix_eqn_trv_rpn_print( libxsmm_matrix_eqns[idx]->eqn_root );
  printf("\n\n");
}


