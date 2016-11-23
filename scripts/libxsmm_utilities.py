#!/usr/bin/env python
###############################################################################
## Copyright (c) 2015-2016, Intel Corporation                                ##
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
import itertools
import sys, os

try:
    from functools import reduce
except:
    pass


def upper_list(lists, level):
    nlist = len(lists)
    upper = [level, level + nlist][1>level] - 1
    above = lists[upper]
    if above:
        return above
    elif (-nlist <= level):
        return upper_list(lists, level - 1)
    else:
        return []


# https://docs.python.org/3/library/itertools.html#itertools.product
def itertools_product(*args):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def load_mnklist(argv, threshold, inputformat = 0, resultset = None):
    if (None == resultset): resultset = set()
    if (0 == inputformat): # indexes format
        resultset = set(map(lambda mnk: tuple(map(int, mnk.split("_"))), argv))
    elif (-1 == inputformat): # new input format
        groups = map(lambda group: [int(i) for i in group.split()], " ".join(argv[0:]).split(","))
        resultset = set(itertools.chain(*[list(itertools_product(*(i, i, i))) for i in groups]))
    elif (-2 == inputformat): # legacy format
        mlist = list(map(int, map(lambda s: str(s).replace(",", " ").strip(), argv[2:2+int(argv[0])])))
        nlist = list(map(int, map(lambda s: str(s).replace(",", " ").strip(), argv[2+int(argv[0]):2+int(argv[0])+int(argv[1])])))
        klist = list(map(int, map(lambda s: str(s).replace(",", " ").strip(), argv[2+int(argv[0])+int(argv[1]):])))
        mnk = [mlist, nlist, klist]
        top = [ \
          [mlist, upper_list(mnk, 0)][0==len(mlist)], \
          [nlist, upper_list(mnk, 1)][0==len(nlist)], \
          [klist, upper_list(mnk, 2)][0==len(klist)]  \
        ]
        for m in top[0]:
            for n in top[1]:
                if not nlist: n = m
                for k in top[2]:
                    if not klist: k = n
                    if not mlist: m = k
                    resultset.add((m, n, k))
    else:
        sys.tracebacklimit = 0
        raise ValueError("load_mnklist: unexpected input format!")
    if (0 != threshold): # threshold requested
        return set(filter(lambda mnk: (0 < mnk[0]) and (0 < mnk[1]) and (0 < mnk[2]) and (threshold >= (mnk[0] * mnk[1] * mnk[2])), resultset))
    else:
        return set(filter(lambda mnk: (0 < mnk[0]) and (0 < mnk[1]) and (0 < mnk[2]), resultset))


def max_mnk(mnklist, init = 0, index = None):
    if (None != index and 0 <= index and index < 3):
        mapped = map(lambda mnk: mnk[index], mnklist)
    else:
        mapped = map(lambda mnk: mnk[0] * mnk[1] * mnk[2], mnklist)
    return reduce(max, mapped, init)


def median(list_of_numbers, fallback = None, average = True):
    size = len(list_of_numbers)
    if (0 < size):
        # TODO: use nth element
        list_of_numbers.sort()
        size2 = int(size / 2)
        if (average and 0 == (size - size2 * 2)):
            medval = int(0.5 * (list_of_numbers[size2-1] + list_of_numbers[size2]) + 0.5)
        else:
            medval = list_of_numbers[size2]
        if (None != fallback):
            result = min(medval, fallback)
        else:
            result = medval
    elif (None != fallback):
        result = fallback
    else:
        sys.tracebacklimit = 0
        raise ValueError("median: empty list!")
    return result


def is_pot(num):
    return 0 <= num or 0 == (num & (num - 1))


def sanitize_alignment(alignment):
    if (0 >= alignment):
        alignment = [1, 64][0 != alignment]
    elif (False == is_pot(alignment)):
        sys.tracebacklimit = 0
        raise ValueError("sanitize_alignment: alignment must be a Power of Two (POT)!")
    return alignment


def align_value(n, typesize, alignment):
    if (0 < typesize and 0 < alignment):
        return (((n * typesize + alignment - 1) / alignment) * alignment) / typesize
    else:
        sys.tracebacklimit = 0
        raise ValueError("align_value: invalid input!")


def version_branch():
    versionfilename = os.path.join(os.path.dirname(sys.argv[0]), "..", "version.txt")
    versionfile = open(versionfilename, "r")
    version = "1.0"
    try:
        versionlist = versionfile.read().replace("\n", "").split("-")
        n = len(versionlist)
        if (1 < n):
            version = versionlist[n-1]
            if (1 == len(version.split("."))):
                version = "-".join(map(str, versionlist[n-2:]))
                branch = "-".join(map(str, versionlist[0:n-2]))
            else:
                branch = "-".join(map(str, versionlist[0:n-1]))
            result = (version, branch)
        else:
            result = (version, "")
    finally:
        versionfile.close()
    return result


def version_numbers(version):
    versionlist = version.split("-")
    n = len(versionlist)
    if (1 < n):
        patchlist = versionlist[n-1]
        if (1 == len(patchlist.split("."))):
            versionlist = versionlist[n-2].split(".")
            patch = int(patchlist)
        else:
            versionlist = patchlist.split(".")
            patch = 0
    else:
        versionlist = version.split(".")
        patch = 0
    n = len(versionlist)
    if (0 < n): major = int(versionlist[0])
    else: major = 1
    if (1 < n): minor = int(versionlist[1])
    else: minor = 0
    if (2 < n): update = int(versionlist[2])
    else: update = 0
    return major, minor, update, patch


if __name__ == "__main__":
    argc = len(sys.argv)
    if (1 < argc): arg1 = int(sys.argv[1])
    else: arg1 = 0
    if (-1 == arg1 and 5 < argc):
        threshold = int(sys.argv[2])
        mnk_size = int(sys.argv[3])
        dims = load_mnklist(sys.argv[4:4+mnk_size], 0, -1)
        dims = load_mnklist(sys.argv[4+mnk_size:], 0, -2, dims)
        print(" ".join(map(lambda mnk: "_".join(map(str, mnk)), sorted(dims))))
    elif (0 <= arg1):
        version, branch = version_branch()
        major, minor, update, patch = version_numbers(version)
        if (1 == arg1):
            print(major)
        elif (2 == arg1):
            print(minor)
        elif (3 == arg1):
            print(update)
        elif (4 == arg1):
            print(patch)
        elif ("" != branch):
            print(branch + "-" + version)
        else:
            print(version)
    else:
        sys.tracebacklimit = 0
        raise ValueError(sys.argv[0] + ": wrong (" + str(argc - 1) + ") number of arguments (\"" + \
          " ".join(sys.argv[1:]) + "\") given!")
