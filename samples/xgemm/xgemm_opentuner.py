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
import inspect
import json
import time
import math
import sys
import os
import re

try:
    here = os.path.dirname(inspect.getfile(inspect.currentframe()))
    scripts = os.path.realpath(os.path.join(here, "..", "..", "scripts"))
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import libxsmm_utilities
except ImportError:
    pass


class XgemmTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        self.dimset = libxsmm_utilities.load_mnklist(self.args.mnk, 0, -1)
        self.granularity = 1
        assert(0 < self.granularity)
        m_max = (64 + self.granularity - 1) / self.granularity
        n_max = (256 + self.granularity - 1) / self.granularity
        k_max = (256 + self.granularity - 1) / self.granularity
        m_param = IntegerParameter("M", self.granularity, m_max)
        n_param = IntegerParameter("N", self.granularity, n_max)
        k_param = IntegerParameter("K", self.granularity, k_max)
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(m_param)
        manipulator.add_parameter(n_param)
        manipulator.add_parameter(k_param)
        return manipulator

    def seed_configurations(self):
        m_seed = self.args.m
        n_seed = [self.args.n, m_seed][0 == self.args.n]
        k_seed = [self.args.k, n_seed][0 == self.args.k]
        if 0 == m_seed or 0 == n_seed or 0 == k_seed:
            return []
        else:
            return [{"M": (m_seed + self.granularity - 1) / self.granularity,
                     "N": (n_seed + self.granularity - 1) / self.granularity,
                     "K": (k_seed + self.granularity - 1) / self.granularity}]

    def objective(self):
        return opentuner.search.objective.MaximizeAccuracyMinimizeSize()

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data
        run_cmd = (
            "CHECK=0"
            " LIBXSMM_TGEMM_M=" + str(self.granularity * cfg["M"]) +
            " LIBXSMM_TGEMM_N=" + str(self.granularity * cfg["N"]) +
            " LIBXSMM_TGEMM_K=" + str(self.granularity * cfg["K"]) +
            " ./xgemm.sh")
        geoperf = 0  # geometric mean
        compensation = 0  # see Kahan
        for dims in self.dimset:
            run_result = self.call_program(
                run_cmd + " " + " ".join(map(str, dims)))
            assert(run_result["returncode"] == 0)
            match = re.search(
                "\\s*LIBXSMM:\\s+([0-9]+(\\.[0-9]*)*)",
                str(run_result["stdout"]))
            assert(match is not None)
            gflops = float(match.group(1))
            assert(0 < gflops)
            kha = math.log(gflops) - compensation
            khb = geoperf + kha
            compensation = (khb - geoperf) - kha
            geoperf = khb
        geoperf = math.exp(geoperf / len(self.dimset))
        geotime = 1000000.0 / geoperf
        mnk = (self.granularity**3) * cfg["M"] * cfg["N"] * cfg["K"]
        return Result(time=geotime, accuracy=geoperf, size=mnk)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        matrices = (  # collects requested matrix shapes into string
            "-".join(map(str, map(lambda mnk: "x".join(
                     map(str, mnk)), self.dimset))))
        filename = "xgemm-" + matrices + time.strftime(
                   "-%Y%m%d-%H%M%S") + ".json"
        print("Optimal block size written to " + filename +
              ": ", configuration.data)
        # self.manipulator().save_to_file(configuration.data, filename)
        with open(filename, 'w') as fd:
            json.dump(configuration.data, fd)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    argparser.add_argument(
        "mnk", nargs="*", default=["1024,1280,1536,1792"],
        help="Set of MNK parameters to be tuned")
    argparser.add_argument(
        "-m", "--initial-m", type=int, default=0, nargs='?',
        dest="m", help="Initial tile size (M)")
    argparser.add_argument(
        "-n", "--initial-n", type=int, default=0, nargs='?',
        dest="n", help="Initial tile size (N)")
    argparser.add_argument(
        "-k", "--initial-k", type=int, default=0, nargs='?',
        dest="k", help="Initial tile size (K)")
    XgemmTuner.main(argparser.parse_args())
