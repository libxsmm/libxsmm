#!/usr/bin/env python3
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################
#
# This script is based on OpenTuner's tutorial:
# "Optimizing Block Matrix Multiplication".
#
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
import random
import json
import time
import sys
import re


class MatcopyTune(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        self.mintilesize = 2
        self.granularity = 1
        assert(0 < self.granularity)
        minsize = max(self.mintilesize / self.granularity, 1)
        maxsize = minsize + self.granularity
        m_max = max(min(self.args.maxm, self.args.end), maxsize)
        n_max = max(min(self.args.maxn, self.args.end), maxsize)
        m_max = (m_max + self.granularity - 1) / self.granularity
        n_max = (n_max + self.granularity - 1) / self.granularity
        m_param = IntegerParameter("M", minsize, m_max)
        n_param = IntegerParameter("N", minsize, n_max)
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(m_param)
        manipulator.add_parameter(n_param)
        return manipulator

    def seed_configurations(self):
        m_seed = [self.args.n, self.args.m][0 != self.args.m]
        n_seed = [self.args.m, self.args.n][0 != self.args.n]
        if 0 == m_seed or 0 == n_seed:
            return []
        else:
            return [{"M": max(m_seed, self.mintilesize),
                     "N": max(n_seed, self.mintilesize)}]

    def objective(self):
        return opentuner.search.objective.MaximizeAccuracyMinimizeSize()

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data
        nruns = max(self.args.nruns, 1)
        begin = max(self.args.begin, self.mintilesize)
        end = max(self.args.end, self.mintilesize)
        m = random.randint(begin, end)
        n = random.randint(begin, end)
        if (self.args.tight):
            ldi = ldo = m
        else:
            ldi = max(random.randint(begin, end), m)
            ldo = max(random.randint(begin, end), m)
        kind = ["COPY", "ZERO"][self.args.zero]
        run_cmd = (
            "CHECK=0 " +  # no checks and only LIBXSMM measurement
            ["ZERO=0", "ZERO=1"][self.args.zero] +
            " LIBXSMM_M" + kind + "_M=" + str(self.granularity * cfg["M"]) +
            " LIBXSMM_M" + kind + "_N=" + str(self.granularity * cfg["N"]) +
            " ./matcopyf "
            + str(m) + " " + str(n) + " " + str(ldi) + " " + str(ldo) + " "
            + str(nruns)) + " " + str(self.args.nmb)
        run_result = self.call_program(run_cmd)
        if (0 == run_result["returncode"]):
            match = re.search(
                "LIBXSMM \\(" + kind.lower() +
                "\\):\\s+([0-9]+(\\.[0-9]*)*)",
                str(run_result["stdout"]))
            assert(match is not None)
            bandwidth = float(match.group(1))
            assert(0 < bandwidth)
            kernelsize = (self.granularity**2) * cfg["M"] * cfg["N"]
            return Result(time=1/bandwidth,
                          accuracy=bandwidth,
                          size=kernelsize)
        else:
            sys.tracebacklimit = 0
            raise RuntimeError("Execution failed for \"" + run_cmd + "\"!")

    def save_final_config(self, configuration):
        """
        called at the end of tuning
        """
        filename = (
            "matcopy-"
            + str(max(self.args.begin, 1)) + "_"
            + str(max(self.args.end,   1))
            + ["_", "_zero_"][self.args.zero]
            + ["_", "_tight_"][self.args.tight]
            + str(max(self.args.nruns, 1)) + "_"
            + str(self.args.nmb) +
            time.strftime("-%Y%m%d-%H%M%S") + ".json")
        print("Optimal block size written to " + filename +
              ": ", configuration.data)
        # self.manipulator().save_to_file(configuration.data, filename)
        with open(filename, 'w') as fd:
            json.dump(configuration.data, fd)


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    argparser = opentuner.default_argparser()
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparser.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    argparser.add_argument(
        "begin", type=int,
        help="Begin of the range (min. M and N)")
    argparser.add_argument(
        "end", type=int,
        help="End of the range (exclusive)")
    argparser.add_argument(
        "m", type=int, default=0, nargs='?',
        help="Initial tile size (M)")
    argparser.add_argument(
        "n", type=int, default=0, nargs='?',
        help="Initial tile size (N)")
    argparser.add_argument(
        "nruns", type=int, default=1, nargs='?',
        help="Number of experiments per epoch")
    argparser.add_argument(
        "nmb", type=int, default=512, nargs='?',
        help="Problem size (MB)")
    argparser.add_argument(
        "maxm", type=int, default=160, nargs='?',
        help="Max. tile size (M)")
    argparser.add_argument(
        "maxn", type=int, default=160, nargs='?',
        help="Max. tile size (N)")
    argparser.add_argument(
        "zero", type=str2bool, nargs='?',
        const=True, default=False,
        help="Zeroing instead of copy")
    argparser.add_argument(
        "tight", type=str2bool, nargs='?',
        const=True, default=True,
        help="Use tight leading dimension")
    MatcopyTune.main(argparser.parse_args())
