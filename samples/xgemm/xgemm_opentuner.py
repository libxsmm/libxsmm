#!/usr/bin/env python
###############################################################################
# Copyright (c) 2017, Intel Corporation                                       #
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
except:
    pass


class XgemmTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        self.granularity = 1
        assert(0 < self.granularity)
        max_m = (256 + self.granularity - 1) / self.granularity
        max_n = (256 + self.granularity - 1) / self.granularity
        max_k = (256 + self.granularity - 1) / self.granularity
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
          IntegerParameter("M", self.granularity, max_m))
        manipulator.add_parameter(
          IntegerParameter("N", self.granularity, max_n))
        manipulator.add_parameter(
          IntegerParameter("K", self.granularity, max_k))
        return manipulator

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
            " LIBXSMM_GEMM_M=" + str(self.granularity * cfg["M"]) +
            " LIBXSMM_GEMM_N=" + str(self.granularity * cfg["N"]) +
            " LIBXSMM_GEMM_K=" + str(self.granularity * cfg["K"]) +
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
    XgemmTuner.main(argparser.parse_args())
