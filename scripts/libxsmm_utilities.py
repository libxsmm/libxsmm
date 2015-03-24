#! /usr/bin/env python
###############################################################################
## Copyright (c) 2013-2015, Intel Corporation                                ##
## All rights reserved.                                                      ##
##                                                                           ##
## Redistribution and use in source and binary forms, with or without        ##
## modification, are permitted provided that the following conditions        ##
## are met:                                                                  ##
## 1. Redistributions of source code must retain the above copyright         ##
##    notice, this list of conditions and the following disclaimer.          ##
## 2. Redistributions in binary form must reproduce the above copyright      ##
##    notice, this list of conditions and the following disclaimer in the    ##
##    documentation and/or other materials provided with the distribution.   ##
## 3. Neither the name of the copyright holder nor the names of its          ##
##    contributors may be used to endorse or promote products derived        ##
##    from this software without specific prior written permission.          ##
##                                                                           ##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       ##
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         ##
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     ##
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      ##
## HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    ##
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  ##
## TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    ##
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    ##
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      ##
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        ##
## SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              ##
###############################################################################
## Hans Pabst (Intel Corp.)
###############################################################################
import sys


def is_pot(num):
    return 0 <= num or 0 == (num & (num - 1))


def make_typeflag(Real):
    return ["s", "d"]["float" != Real]


def make_typepfix(Real):
    return ["", "d"]["float" != Real]


def upper_list(lists, level):
    upper = [level, level + len(lists)][1>level] - 1
    above = lists[upper]
    if above:
        return above
    else:
        return upper_list(lists, level - 1)


def make_mnklist(mlist, nlist, klist):
    mnk = [mlist, nlist, klist]
    top = [ \
      [mlist, upper_list(mnk, 0)][0==len(mlist)], \
      [nlist, upper_list(mnk, 1)][0==len(nlist)], \
      [klist, upper_list(mnk, 2)][0==len(klist)]  \
    ]
    result = set()
    for m in top[0]:
        for n in top[1]:
            if not nlist: n = m
            for k in top[2]:
                if not klist: k = n
                if not mlist: m = k
                result.add((m, n, k))
    mnklist = list(result)
    mnklist.sort()
    return mnklist


def make_mlist(mnklist):
    return map(lambda mnk: mnk[0], mnklist)


def make_nlist(mnklist):
    return map(lambda mnk: mnk[1], mnklist)


def make_klist(mnklist):
    return map(lambda mnk: mnk[2], mnklist)


def load_mlist(argv):
    begin = 3; end = begin + int(argv[1])
    if (begin > end or end > len(argv)):
        raise ValueError("load_mlist: wrong number of elements!")
    return map(int, argv[begin:end])


def load_nlist(argv):
    begin = 3 + int(argv[1]); end = begin + int(argv[2])
    if (begin > end or end > len(argv)):
        raise ValueError("load_nlist: wrong number of elements!")
    return map(int, argv[begin:end])


def load_klist(argv):
    begin = 3 + int(argv[1]) + int(argv[2])
    if (begin > len(argv)):
        raise ValueError("load_klist: wrong number of elements!")
    return map(int, argv[begin:])


def load_mnklist(argv):
    return make_mnklist(load_mlist(argv), load_nlist(argv), load_klist(argv))


def load_mnksize(argv):
    msize = int(argv[1])
    nsize = int(argv[2])
    mnsize = msize + nsize + 2
    mnksize = len(argv) - 1
    if (mnsize > mnksize):
        raise ValueError("load_mnksize: malformed index list!")
    return (msize, nsize, mnksize - mnsize)


def max_mnk(mnklist, init = 0, index = None):
    return reduce(max, map(lambda mnk: \
      mnk[index] if (None != index and 0 <= index and index < 3) \
      else (mnk[0] * mnk[1] * mnk[2]), mnklist), init)


if __name__ == '__main__':
    if (3 < len(sys.argv)):
        mnklist = load_mnklist(sys.argv)
        print " ".join(map(lambda mnk: "_".join(map(str, mnk)), mnklist))
    else:
        raise ValueError(sys.argv[0] + ": wrong number of arguments!")
