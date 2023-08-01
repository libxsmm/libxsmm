#!/usr/bin/env python3
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################
import itertools
import operator
import inspect
import ctypes
import ctypes.util
import sys
import os

try:
    from functools import reduce
except ImportError:
    pass


def upper_list(lists, level):
    nlist = len(lists)
    upper = [level, level + nlist][1 > level] - 1
    above = lists[upper]
    if above:
        return above
    elif -nlist <= level:
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
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def load_mnklist(argv, threshold, inputformat=0, resultset=None):
    if resultset is None:
        resultset = set()
    if 0 == inputformat:  # indexes format
        resultset = set(map(lambda mnk: tuple(map(int, mnk.split("_"))), argv))
    elif -1 == inputformat:  # new input format
        groups = map(
            lambda group: [int(i) for i in group.split()],
            " ".join(argv[0:]).split(","),
        )
        resultset = set(
            itertools.chain(
                *[list(itertools_product(*(i, i, i))) for i in groups]
            )
        )
    elif -2 == inputformat:  # legacy format
        mlist = list(
            map(
                int,
                map(
                    lambda s: str(s).replace(",", " ").strip(),
                    argv[2 : 2 + int(argv[0])],  # noqa: E203
                ),
            )
        )
        nlist = list(
            map(
                int,
                map(
                    lambda s: str(s).replace(",", " ").strip(),
                    argv[
                        2
                        + int(argv[0]) : 2  # noqa: E203
                        + int(argv[0])
                        + int(argv[1])  # noqa: E203
                    ],
                ),
            )
        )
        klist = list(
            map(
                int,
                map(
                    lambda s: str(s).replace(",", " ").strip(),
                    argv[2 + int(argv[0]) + int(argv[1]) :],  # noqa: E203
                ),
            )
        )
        mnk = [mlist, nlist, klist]
        top = [
            [mlist, upper_list(mnk, 0)][0 == len(mlist)],
            [nlist, upper_list(mnk, 1)][0 == len(nlist)],
            [klist, upper_list(mnk, 2)][0 == len(klist)],
        ]
        for m in top[0]:
            for n in top[1]:
                if not nlist:
                    n = m
                for k in top[2]:
                    if not klist:
                        k = n
                    if not mlist:
                        m = k
                    resultset.add((m, n, k))
    else:
        sys.tracebacklimit = 0
        raise ValueError("load_mnklist: unexpected input format!")
    if 0 != threshold:  # threshold requested
        return set(
            filter(
                lambda mnk: (0 < mnk[0])
                and (0 < mnk[1])
                and (0 < mnk[2])
                and (threshold >= (mnk[0] * mnk[1] * mnk[2])),
                resultset,
            )
        )
    else:
        return set(
            filter(
                lambda mnk: (0 < mnk[0]) and (0 < mnk[1]) and (0 < mnk[2]),
                resultset,
            )
        )


def max_mnk(mnklist, init=0, index=None):
    if index is not None and 0 <= index and index < 3:
        mapped = map(lambda mnk: mnk[index], mnklist)
    else:
        mapped = map(lambda mnk: mnk[0] * mnk[1] * mnk[2], mnklist)
    return reduce(max, mapped, init)


def median(list_of_numbers, fallback=None, average=True):
    size = len(list_of_numbers)
    if 0 < size:
        # TODO: use nth element
        list_of_numbers.sort()
        size2 = int(size / 2)
        if average and 0 == (size - size2 * 2):
            medval = int(
                0.5 * (list_of_numbers[size2 - 1] + list_of_numbers[size2])
                + 0.5
            )
        else:
            medval = list_of_numbers[size2]
        if fallback is not None:
            result = min(medval, fallback)
        else:
            result = medval
    elif fallback is not None:
        result = fallback
    else:
        sys.tracebacklimit = 0
        raise ValueError("median: empty list!")
    return result


def is_pot(num):
    return 0 <= num or 0 == (num & (num - 1))


def sanitize_alignment(alignment):
    if 0 >= alignment:
        alignment = [1, 64][0 != alignment]
    elif not is_pot(alignment):
        sys.tracebacklimit = 0
        raise ValueError(
            "sanitize_alignment: alignment must be a Power of Two (POT)!"
        )
    return alignment


def align_value(n, typesize, alignment):
    if 0 < typesize and 0 < alignment:
        return (
            ((n * typesize + alignment - 1) / alignment) * alignment
        ) / typesize
    else:
        sys.tracebacklimit = 0
        raise ValueError("align_value: invalid input!")


def version_branch_from_file(version_filepath):
    version_file = open(version_filepath, "r")
    version, branch, sep = "1.0", "", "-"
    try:
        version_list, n = version_file.read().replace("\n", "").split(sep), 0
        for word in version_list:
            if not reduce(
                operator.and_,
                (subword.isdigit() for subword in word.split(".")),
                True,
            ):
                branch += [sep + word, word][0 == n]
                n += 1
            else:
                break
        version = sep.join(version_list[n:])
    finally:
        version_file.close()
    return (version, branch)


def version_numbers(version, branch=None):
    version_list = version.split("-")
    if (
        0 < len(version_list)
        and version_list[0]
        and not version_list[0][0].isdigit()
    ):
        vbranch = version_list[0]
    else:
        vbranch = "main"
    if branch is None or vbranch == branch:
        minor = update = patch = 0
        major = 1
        n = len(version_list)
        if 1 < n:
            patch_list = version_list[n - 1]
            if 1 == len(patch_list.split(".")):
                version_list = version_list[n - 2].split(".")
                if version_list != [vbranch]:
                    patch = int(patch_list)
                else:
                    major = int(patch_list)
            else:
                version_list = patch_list.split(".")
        else:
            version_list = version.split(".")
        n = len(version_list)
        try:
            if 0 < n:
                major = int(version_list[0])
            if 1 < n:
                minor = int(version_list[1])
            if 2 < n:
                update = int(version_list[2])
        except ValueError:
            # if 1 == n: major = 0
            pass
    else:
        major = minor = update = patch = -1
    return major, minor, update, patch


def version_branch(max_strlen=-1):
    version_filename = "version.txt"
    filepath_default = os.path.realpath(
        os.path.join(
            os.path.dirname(inspect.getfile(inspect.currentframe())),
            "..",
            version_filename,
        )
    )
    filepath_local = os.path.realpath(version_filename)  # local version file
    realversion, branch = version_branch_from_file(filepath_default)
    version = realversion
    out_of_tree = filepath_default != filepath_local
    if out_of_tree and os.path.isfile(filepath_local):
        local, ignored = version_branch_from_file(filepath_local)
        if version_numbers(realversion) < version_numbers(local):
            version = local
    if 0 < max_strlen:
        start = int(max_strlen / 3)
        cut = max(
            branch.rfind("-", start, max_strlen),
            branch.rfind("_", start, max_strlen),
            branch.rfind(".", start, max_strlen),
        )
        if start < cut:
            branch = branch[0:cut]
        else:
            branch = branch[0:max_strlen]
    return (version, branch, realversion)


def libxsmm_target_arch():
    libpath = os.path.join(os.path.dirname(__file__), "..", "lib")
    uname = os.uname()
    oname = uname.sysname if not isinstance(uname, tuple) else uname[0]
    if "Darwin" == oname:
        os.environ["DYLD_LIBRARY_PATH"] = libpath
        libext = ".dylib"
    else:
        os.environ["LD_LIBRARY_PATH"] = libpath
        libext = ".so"
    os.environ["LIBXSMM_VERBOSE"] = "0"
    xsmmnoblas = (
        "libxsmmnoblas" + libext
        if os.path.exists(os.path.join(libpath, "libxsmmnoblas" + libext))
        else ctypes.util.find_library("xsmmnoblas")
    )
    xsmm = (
        "libxsmm" + libext
        if os.path.exists(os.path.join(libpath, "libxsmm" + libext))
        else ctypes.util.find_library("xsmm")
    )
    target = "generic"
    try:
        libxsmm = ctypes.CDLL(
            os.path.join(libpath, xsmmnoblas), mode=ctypes.RTLD_GLOBAL
        )
        libxsmm = ctypes.CDLL(
            os.path.join(libpath, xsmm), mode=ctypes.RTLD_GLOBAL
        )
        libxsmm_get_target_arch = libxsmm.libxsmm_get_target_arch
        libxsmm_get_target_arch.restype = ctypes.c_char_p
        target = libxsmm_get_target_arch().decode("ascii")
    except:  # noqa: E722
        sys.stderr.write(
            "WARNING: falling back to {0} target.\n".format(target)
        )
        pass
    return target


if __name__ == "__main__":
    argc = len(sys.argv)
    if 1 < argc:
        arg1 = int(sys.argv[1])
    else:
        arg1 = 0
    if -1 >= arg1:
        if 5 < argc:
            # threshold = int(sys.argv[2])
            mnk_size = int(sys.argv[3])
            dims = load_mnklist(
                sys.argv[4 : 4 + mnk_size], 0, -1  # noqa: E203
            )  # noqa: E203
            dims = load_mnklist(
                sys.argv[4 + mnk_size :], 0, -2, dims  # noqa: E203
            )  # noqa: E203
            mnklist = map(lambda mnk: "_".join(map(str, mnk)), sorted(dims))
            print(" ".join(mnklist))
        elif 3 == argc:
            major, minor, update, patch = version_numbers(
                sys.argv[2], "release"
            )
            print(["0", "1"][0 == patch])
    else:  # 0 <= arg1
        if 0 == arg1 and 3 == argc:
            major, minor, update, patch = version_numbers(sys.argv[2])
            print(major)  # soname version
        elif 0 == arg1 and 1 == argc:
            print(libxsmm_target_arch())
        else:
            version, branch, realversion = version_branch()
            major, minor, update, patch = version_numbers(version)
            if 1 == arg1:
                print(major)
            elif 2 == arg1:
                print(minor)
            elif 3 == arg1:
                print(update)
            elif 4 == arg1:
                print(patch)
            elif 5 == arg1:
                print(realversion)
            elif "" != branch:
                print("{0}-{1}".format(branch, realversion))
            else:
                print(realversion)
