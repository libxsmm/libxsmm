#!/usr/bin/env python
###############################################################################
# Copyright (c) 2017-2018, Intel Corporation                                  #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions          #
# are met:                                                                    #
# 1. Redistributions of source code must retain the above copyright           #
#    notice, this list of conditions and the following disclaimer.            #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR       #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT        #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
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
        self.granularity = 1
        assert(0 < self.granularity)
        m_max = (256 + self.granularity - 1) / self.granularity
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
        m_seed = [self.args.n, self.args.m][0 != self.args.m]
        k_seed = [self.args.m, self.args.k][0 != self.args.k]
        n_seed = [self.args.k, self.args.n][0 != self.args.n]
        if 0 == m_seed or 0 == n_seed or k_seed:
            return []
        else:
            return [{"M": m_seed, "N": n_seed, "K": k_seed}]

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
        dimset = libxsmm_utilities.load_mnklist(self.args.mnk, 0, -1)
        geoperf = 0  # geometric mean
        compensation = 0  # see Kahan
        for dims in dimset:
            run_result = self.call_program(
                run_cmd + " " + " ".join(map(str, dims)))
            assert(run_result["returncode"] == 0)
            match = re.search(
                "\s*LIBXSMM:\s+([0-9]+(\.[0-9]*)*)",
                run_result["stdout"])
            assert(match is not None)
            gflops = float(match.group(1))
            assert(0 < gflops)
            kha = math.log(gflops) - compensation
            khb = geoperf + kha
            compensation = (khb - geoperf) - kha
            geoperf = khb
        geoperf = math.exp(geoperf / len(dimset))
        geotime = 1000000.0 / geoperf
        mnk = (self.granularity**3) * cfg["M"] * cfg["N"] * cfg["K"]
        return Result(time=geotime, accuracy=geoperf, size=mnk)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        dimset = libxsmm_utilities.load_mnklist(self.args.mnk, 0, -1)
        matrices = (  # collects requested matrix shapes into string
            "-".join(map(str, map(lambda mnk: "x".join(
                     map(str, mnk)), dimset))))
        filename = "xgemm-" + matrices + time.strftime(
                   "-%Y%m%d-%H%M%S") + ".json"
        print("Optimal block size written to " + filename +
              ": ", configuration.data)
        self.manipulator().save_to_file(configuration.data, filename)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    argparser.add_argument(
        "mnk", metavar="N", nargs="*", default=["1024,1280,1536,1792"],
        help="Set of MNK parameters to be tuned")
    argparser.add_argument(
        "m", type=int, default=0, nargs='?',
        help="Initial tile size (M)")
    argparser.add_argument(
        "n", type=int, default=0, nargs='?',
        help="Initial tile size (N)")
    argparser.add_argument(
        "k", type=int, default=0, nargs='?',
        help="Initial tile size (K)")
    XgemmTuner.main(argparser.parse_args())
