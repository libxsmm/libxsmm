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
import math
import re


class XgemmTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        self.granularity = 4
        assert(0 < self.granularity)
        max_m = (160 + self.granularity - 1) / self.granularity
        max_n = (64 + self.granularity - 1) / self.granularity
        max_k = (96 + self.granularity - 1) / self.granularity
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
          IntegerParameter("M", self.granularity, max_m))
        manipulator.add_parameter(
          IntegerParameter("N", self.granularity, max_n))
        manipulator.add_parameter(
          IntegerParameter("K", self.granularity, max_k))
        return manipulator

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data
        run_cmd = (
            "TASKS=1"
            " LIBXSMM_M=" + str(self.granularity * cfg["M"]) +
            " LIBXSMM_N=" + str(self.granularity * cfg["N"]) +
            " LIBXSMM_K=" + str(self.granularity * cfg["K"]) +
            " ./xgemm.sh")

        geoperf = 0  # geometric mean
        compensation = 0  # see Kahan
        sizelist = [1000, 2000, 3000, 4000]
        for size in sizelist:
            run_result = self.call_program(run_cmd + " " + str(size))
            assert(run_result["returncode"] == 0)
            match = re.search(
                "\s*LIBXSMM:\s+([0-9]+(\.[0-9]*)*)",
                run_result["stdout"])
            assert(match is not None)
            kha = math.log(float(match.group(1))) - compensation
            khb = geoperf + kha
            compensation = (khb - geoperf) - kha
            geoperf = khb
        geoperf = math.exp(geoperf / len(sizelist))
        geotime = 1000000.0 / geoperf

        return Result(time=geotime)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        filename = "xgemm_opentuner.json"
        print("Optimal block size written to " +
              filename + ": ", configuration.data)
        self.manipulator().save_to_file(configuration.data, filename)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    XgemmTuner.main(argparser.parse_args())
