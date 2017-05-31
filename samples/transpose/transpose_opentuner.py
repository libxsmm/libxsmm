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
import time
import sys
import re


class TransposeTune(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        self.granularity = 1
        assert(0 < self.granularity)
        max_m = (160 + self.granularity - 1) / self.granularity
        max_n = (160 + self.granularity - 1) / self.granularity
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
          IntegerParameter("M", self.granularity, max_m))
        manipulator.add_parameter(
          IntegerParameter("N", self.granularity, max_n))
        return manipulator

    def objective(self):
        return opentuner.search.objective.MaximizeAccuracyMinimizeSize()

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data
        nruns = max(self.args.nruns, 1)
        end = max(self.args.end, 0)
        run_cmd = (
            "CHECK=0"
            " LIBXSMM_TRANS_M=" + str(self.granularity * cfg["M"]) +
            " LIBXSMM_TRANS_N=" + str(self.granularity * cfg["N"]) +
            " ./transpose.sh o" + " " + str(end) + " " + str(end) +
            " " + str(end) + " " + str(end) + " " + str(nruns) +
            " " + str(max(self.args.begin, 0)))
        run_result = self.call_program(run_cmd)
        if (0 == run_result["returncode"]):
            match = re.search(
                "\s*duration:\s+([0-9]+(\.[0-9]*)*)",
                run_result["stdout"])
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
            "transpose-" + str(max(self.args.begin, 0)) +
            "_" + str(max(self.args.end,   0)) +
            "_" + str(max(self.args.nruns, 1)) +
            time.strftime("-%Y%m%d-%H%M%S") + ".json")
        print("Optimal block size written to " + filename +
              ": ", configuration.data)
        self.manipulator().save_to_file(configuration.data, filename)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    argparser.add_argument(
        "begin", type=int, default=1024,
        help="Begin of the range")
    argparser.add_argument(
        "end", type=int, default=2048,
        help="End of the range")
    argparser.add_argument(
        "nruns", type=int, default=100,
        help="Number of runs")
    TransposeTune.main(argparser.parse_args())
