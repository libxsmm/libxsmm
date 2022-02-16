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
#
# This script is based on OpenTuner's tutorial:
# "Optimizing Block Matrix Multiplication".
#
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
import json
import time
import sys
import re


class TransposeTune(MeasurementInterface):
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
        run_cmd = (
            "CHECK=-1"  # repeatable runs
            " LIBXSMM_TCOPY_M=" + str(self.granularity * cfg["M"]) +
            " LIBXSMM_TCOPY_N=" + str(self.granularity * cfg["N"]) +
            " ./transpose.sh o" + " " + str(end) + " " + str(end) +
            " " + str(end) + " " + str(end) + " " + str(nruns) +
            " -" + str(begin))
        run_result = self.call_program(run_cmd)
        if (0 == run_result["returncode"]):
            match = re.search(
                "\\s*duration:\\s+([0-9]+(\\.[0-9]*)*)",
                str(run_result["stdout"]))
            assert(match is not None)
            mseconds = float(match.group(1)) / nruns
            assert(0 < mseconds)
            frequency = 1000.0 / mseconds
            kernelsize = (self.granularity**2) * cfg["M"] * cfg["N"]
            return Result(time=mseconds, accuracy=frequency, size=kernelsize)
        else:
            sys.tracebacklimit = 0
            raise RuntimeError("Execution failed for \"" + run_cmd + "\"!")

    def save_final_config(self, configuration):
        """
        called at the end of tuning
        """
        filename = (
            "transpose-" + str(max(self.args.begin, 1)) +
            "_" + str(max(self.args.end,   1)) +
            "_" + str(max(self.args.nruns, 1)) +
            time.strftime("-%Y%m%d-%H%M%S") + ".json")
        print("Optimal block size written to " + filename +
              ": ", configuration.data)
        # self.manipulator().save_to_file(configuration.data, filename)
        with open(filename, 'w') as fd:
            json.dump(configuration.data, fd)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    argparser.add_argument(
        "begin", type=int,
        help="Begin of the range (min. M and N)")
    argparser.add_argument(
        "end", type=int,
        help="End of the range (max. M and N)")
    argparser.add_argument(
        "nruns", type=int, default=100, nargs='?',
        help="Number of experiments per epoch")
    argparser.add_argument(
        "m", type=int, default=0, nargs='?',
        help="Initial tile size (M)")
    argparser.add_argument(
        "n", type=int, default=0, nargs='?',
        help="Initial tile size (N)")
    argparser.add_argument(
        "maxm", type=int, default=160, nargs='?',
        help="Max. tile size (M)")
    argparser.add_argument(
        "maxn", type=int, default=160, nargs='?',
        help="Max. tile size (N)")
    TransposeTune.main(argparser.parse_args())
