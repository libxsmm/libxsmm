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
import matplotlib.pyplot as plot
import statistics
import requests
import argparse
import pathlib
import json
import re


def main(args):
    urlbase = "https://api.buildkite.com/v2/organizations"
    url = f"{urlbase}/{args.organization}/pipelines/{args.pipeline}/builds"
    auth = {"Authorization": f"Bearer {args.token}"} if args.token else None
    params = {"per_page": 100, "page": 1}
    query = args.query.lower()

    try:  # proceeed with cached results in case of an error
        builds = requests.get(url, params=params, headers=auth).json()
    except:  # noqa: E722
        print(f"WARNING: failed to connect to {url}\n")
        builds = None
        pass

    if not builds:
        print("ERROR: token is missing (not authorized)")
        exit(1)
    elif "message" in builds:
        message = builds["message"]
        print(f"ERROR: {message}")
        exit(1)

    try:
        with open(args.filepath, "r") as file:
            database = json.load(file)
    except:  # noqa: E722
        database = dict()
        pass

    latest = 0
    nerrors = 0
    nentries = 0
    for build in builds:
        nbuild = build["number"]
        if latest < nbuild:
            latest = nbuild
        # JSON stores integers as string
        sbuild = str(nbuild)
        if sbuild in database:
            break
        print("|", end="", flush=True)
        for job in (job for job in build["jobs"] if 0 == job["exit_status"]):
            print(".", end="", flush=True)
            log = requests.get(job["log_url"], headers=auth)
            txt = json.loads(log.text)["content"]
            for match in (
                match
                for match in re.finditer(
                    r"^\+\+\+ PERFORMANCE ([\w-]+)([^+]+)*", txt, re.MULTILINE
                )
                if match and match.group(1) and match.group(2)
            ):
                values = [
                    line.group(1)
                    for line in re.finditer(r"([^\n\r]+)", match.group(2))
                    if line and line.group(1)
                ]
                if not any("syntax error" in v for v in values):
                    if sbuild not in database:
                        database[sbuild] = dict()
                    if match.group(1) not in database[sbuild]:
                        database[sbuild][match.group(1)] = dict()
                    if job["name"] not in database[sbuild][match.group(1)]:
                        database[sbuild][match.group(1)][job["name"]] = dict()
                    database[sbuild][match.group(1)][job["name"]] = values
                    nentries = nentries + 1
                else:
                    nerrors = nerrors + 1

    if 0 != nerrors or 0 != nentries:
        print("")
    if 0 != nerrors:
        entries = "entr" + "ies" if 1 != nerrors else "y"
        print(f"Ignored {nerrors} erroneous {entries}!")
    entries = "entr" + "ies" if 1 != nentries else "y"
    print(f"Found {nentries} new {entries}.")
    if 0 != nentries:
        with open(args.filepath, "w") as file:
            json.dump(database, file, indent=2)
            file.write("\n")  # append newline at EOF

    template = database[str(latest)]
    figure, axes = plot.subplots(
        len(template), sharex=True, figsize=(9, 6), dpi=300
    )  # noqa: E501
    i = 0
    for entry in template:
        for q in (q for q in template[entry] if query in q.lower()):
            ylabel = None
            yvalue = []
            for build in (
                build
                for build in database
                if entry in database[build] and q in database[build][entry]
            ):
                for v in database[build][entry][q]:
                    match = re.match(
                        r"(.+)?(^|[\s:=])([+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?)",  # noqa: E501
                        v,
                    )
                    if match and match.group(3):
                        init = match.group(1).strip() if match.group(1) else ""
                        unit = v[match.end(3) :].strip()  # noqa: E203
                        if (args.result.lower() in init.lower()) or (
                            args.result.lower() in unit.lower()
                        ):
                            if not ylabel:
                                ylabel = unit if unit else init
                            yvalue.append(float(match.group(3)))
                            break
                if args.history <= len(yvalue):
                    break
            unit = ylabel if ylabel else args.result
            if 0 < args.median:
                yvs = yvalue[0 : args.median]  # noqa: E203
                geo = statistics.geometric_mean([y for y in yvs if 0 < y])
                label = f"{q} = {int(geo)} {unit}"
            else:
                label = q
            axes[i].plot(yvalue, ".:", label=label)
        axes[i].set_title(entry.upper())
        axes[i].legend()
        i = i + 1
    figure.suptitle(f"Performance History in {unit}", fontsize="x-large")
    figure.gca().invert_xaxis()
    figure.tight_layout()
    figure.savefig(args.filepath.stem + ".png")


if __name__ == "__main__":
    here = pathlib.Path(__file__).absolute().parent
    argparser = argparse.ArgumentParser(
        description="Report results from Continuous Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "-f",
        "--filepath",
        type=pathlib.Path,
        default=here / "tool_report.json",
        help="JSON-database used to cache results",
    )
    argparser.add_argument(
        "-s",
        "--organization",
        type=str,
        default="intel",
        help="Buildkite organization/slug",
    )
    argparser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        default="tpp-libxsmm",
        help="Buildkite pipeline",
    )
    argparser.add_argument(
        "-a",
        "--token",
        type=str,
        help="Authorization token",
    )
    argparser.add_argument(
        "-q",
        "--query",
        type=str,
        default="resnet",
        help="Set of values",
    )
    argparser.add_argument(
        "-r",
        "--result",
        type=str,
        default="ms",
        help="Kind of value",
    )
    argparser.add_argument(
        "-m",
        "--median",
        type=int,
        default=7,
        help="Number of samples",
    )
    argparser.add_argument(
        "-n",
        "--history",
        type=int,
        default=25,
        help="Number of builds",
    )
    args = argparser.parse_args()
    main(args)
