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


def parselog(database, strbuild, jobname, txt, nentries, nerrors):
    for match in (
        match
        for match in re.finditer(
            r"^\+\+\+ PERFORMANCE ([\w-]+)([^+-]+)*", txt, re.MULTILINE
        )
        if match and match.group(1) and match.group(2)
    ):
        values = [
            line.group(1)
            for line in re.finditer(r"([^\n\r]+)", match.group(2))
            if line and line.group(1)
        ]
        if not any("syntax error" in v for v in values):
            if strbuild not in database:
                database[strbuild] = dict()
            if match.group(1) not in database[strbuild]:
                database[strbuild][match.group(1)] = dict()
            if jobname not in database[strbuild][match.group(1)]:
                database[strbuild][match.group(1)][jobname] = dict()
            database[strbuild][match.group(1)][jobname] = values
            nentries = nentries + 1
        else:
            nerrors = nerrors + 1
    return nentries, nerrors


def parseval(string):
    """
    Split "string" into "init", "value", and "unit".
    """
    return re.match(
        r"(.+)?(^|[\s:=])([+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?)",
        string,  # noqa: E501
    )


def matchstr(s1, s2):
    if s1:
        if not re.search(r"\d+$", s1) or not re.search(r"\d+$", s2):
            return s1 in s2
        else:  # avoid matching, e.g. "a12" if "a1" is searched
            return (s1 + ".") in (s2 + ".")
    else:
        return False


def num2int(num):
    return int((num + 0.5) if 0 <= num else (num - 0.5))


def num2str(num):
    return (
        f"$\pm${num}"  # noqa: W605
        if 0 == num
        else (f"+{num}" if 0 < num else f"{num}")
    )


def main(args, argd):
    urlbase = "https://api.buildkite.com/v2/organizations"
    url = f"{urlbase}/{args.organization}/pipelines/{args.pipeline}/builds"
    auth = {"Authorization": f"Bearer {args.token}"} if args.token else None
    params = {"per_page": 100, "page": 1}
    select = args.select.lower().split()
    query = args.query.lower().split()
    smry = args.summary.lower()
    rslt = args.result.lower()
    sdo = 0 < args.mean and smry != rslt
    nerrors = nentries = 0
    match = set()

    try:
        with open(args.filepath, "r") as file:
            database = json.load(file)
    except:  # noqa: E722
        print(f"Create new database {args.filepath}")
        database = dict()
        pass
    latest = max(int(e) for e in database) if database else 0

    if args.infile:
        try:
            with open(args.infile, "r") as file:
                txt = file.read()
        except:  # noqa: E722
            args.infile = None
            pass

    if args.infile:
        outfile = f"{args.infile.stem}.json"
        nentries, nerrors = parselog(
            database, str(latest + 1), args.infile.stem, txt, nentries, nerrors
        )
        if 0 < nentries:
            latest = latest + 1
    else:  # connect to URL
        outfile = args.filepath
        try:  # proceeed with cached results in case of an error
            builds = requests.get(url, params=params, headers=auth).json()
        except:  # noqa: E722
            builds = None
            pass
        if builds and "message" in builds:
            message = builds["message"]
            print(f"ERROR: {message}")
            exit(1)
        elif not args.token:
            print("ERROR: token is missing!")
            exit(1)
        elif not builds:
            print(f"WARNING: failed to connect to {url}.")

        njobs = 0
        while builds:
            # iterate over all builds (latest first)
            for build in builds:
                running = "running" == build["state"]
                nbuild = build["number"]
                # JSON stores integers as string
                strbuild = str(nbuild)
                if not running and strbuild in database:
                    break
                jobs = build["jobs"]
                n = 0
                for job in (job for job in jobs if 0 == job["exit_status"]):
                    if 0 == n:
                        print(f"[{nbuild}]", end="", flush=True)
                    print(".", end="", flush=True)
                    log = requests.get(job["log_url"], headers=auth)
                    txt = json.loads(log.text)["content"]
                    nentries, nerrors = parselog(
                        database, strbuild, job["name"], txt, nentries, nerrors
                    )
                    n = n + 1
                njobs = njobs + n
                if not running and latest < nbuild:
                    latest = nbuild
            if 1 < nbuild:
                params["page"] = params["page"] + 1  # next page
                builds = requests.get(url, params=params, headers=auth).json()
            else:
                builds = None
        if 0 < njobs:
            print("[OK]")

    if 0 != nerrors:
        y = "ies" if 1 != nerrors else "y"
        print(f"Ignored {nerrors} erroneous entr{y}!")
    y = "ies" if 1 != nentries else "y"
    print(f"Found {nentries} new entr{y}.")
    if database:
        database = dict(sorted(database.items(), key=lambda v: int(v[0])))
    if 0 != nentries:
        with open(outfile, "w") as file:
            json.dump(database, file, indent=2)
            file.write("\n")  # append newline at EOF

    template = database[str(latest)] if str(latest) in database else []
    nselect = sum(
        1
        for e in template
        if not select
        or any(matchstr(s, e.lower()) for s in select)  # noqa: E501
    )
    figure, axes = plot.subplots(
        max(nselect, 1), sharex=True, figsize=(9, 6), dpi=300
    )  # noqa: E501
    if 2 > nselect:
        axes = [axes]
    i = 0
    yunit = None
    for entry in (
        e
        for e in template
        if not select
        or any(matchstr(s, e.lower()) for s in select)  # noqa: E501
    ):
        for value in (
            v
            for v in template[entry]
            if not query or any(matchstr(p, v.lower()) for p in query)
        ):
            yvalue = []  # determined by --result
            meanvl = []  # determined by --summary
            sunit = aunit = None
            analyze = dict()
            match.add(value)
            for build in (
                b
                for b in database
                if entry in database[b] and value in database[b][entry]
            ):
                ylabel = slabel = None
                values = database[build][entry][value]
                # match --result primarily against "unit"
                for v in reversed(values):  # match last entry
                    parsed = parseval(v)
                    if parsed and parsed.group(3):
                        unit = v[parsed.end(3) :].strip()  # noqa: E203
                        ulow = unit.lower()
                        if not ylabel and matchstr(rslt, ulow):
                            yvalue.append(float(parsed.group(3)))
                            ylabel = unit
                        if not slabel and sdo and matchstr(smry, ulow):
                            meanvl.append(float(parsed.group(3)))
                            slabel = unit
                # match --result secondary against "init"
                for v in reversed(values):  # match last entry
                    parsed = parseval(v)
                    if parsed and parsed.group(3):
                        init = (
                            parsed.group(1).strip(": ")
                            if parsed.group(1)
                            else ""  # noqa: E501
                        )
                        unit = v[parsed.end(3) :].strip()  # noqa: E203
                        ulab = unit if unit else init
                        ilow = init.lower()
                        if not ylabel and matchstr(rslt, ilow):
                            yvalue.append(float(parsed.group(3)))
                            ylabel = ulab
                        if not slabel and sdo and matchstr(smry, ilow):
                            meanvl.append(float(parsed.group(3)))
                            slabel = ulab
                        if (not aunit or ulab == aunit) and matchstr(
                            args.analyze, ilow
                        ):
                            if init not in analyze:
                                if not aunit:
                                    aunit = ulab
                                analyze[init] = []
                            analyze[init].append(float(parsed.group(3)))

            if yvalue:  # (re-)reverse and trim collected values
                yvalue = yvalue[: -args.history - 1 : -1]  # noqa: E203
            for a in analyze:  # (re-)reverse and trim collected values
                analyze[a] = analyze[a][: -args.history - 1 : -1]  # noqa: E203

            if not yunit:
                yunit = (ylabel if ylabel else args.result).split()[0]
            if 0 < args.mean:
                if meanvl:  # (re-)reverse and trim collected values
                    meanvl = meanvl[: -args.history - 1 : -1]  # noqa: E203
                values = [v for v in (meanvl if meanvl else yvalue) if 0 < v]
                vnew = values[0 : args.mean]  # noqa: E203
                if vnew:
                    if not sunit:
                        sunit = (slabel if slabel else args.result).split()[0]
                    mnew = statistics.geometric_mean(vnew)
                    vold = values[args.mean :]  # noqa: E203
                    label = f"{value} = {num2int(mnew)} {sunit}"
                    if vold:
                        mold = statistics.geometric_mean(vold)
                        perc = num2int(100 * (mnew - mold) / mold)
                        label = f"{label} ({num2str(perc)}%)"

                        if 0 != perc and args.analyze:
                            amax = float("-inf")
                            amin = float("inf")
                            for a in analyze:
                                values = [v for v in analyze[a] if 0 < v]
                                vnew = values[0 : args.mean]  # noqa: E203
                                vold = values[args.mean :]  # noqa: E203
                                if vnew and vold:
                                    anew = statistics.geometric_mean(vnew)
                                    aold = statistics.geometric_mean(vold)
                                    perc = num2int(100 * (anew - aold) / aold)
                                    if perc > amax:
                                        vmax = num2int(anew)
                                        analyze_max = a
                                        amax = perc
                                    elif perc < amin:
                                        vmin = num2int(anew)
                                        analyze_min = a
                                        amin = perc
                            unit = f" {aunit}" if aunit else ""
                            if analyze_min and 0 != vmin and 0 != amin:
                                vlabel = analyze_min.replace(" ", "")
                                label = f"{label} {vlabel}={vmin}{unit} ({num2str(amin)}%)"  # noqa: E501
                            if analyze_max and 0 != vmax and 0 != amax:
                                vlabel = analyze_max.replace(" ", "")
                                label = f"{label} {vlabel}={vmax}{unit} ({num2str(amax)}%)"  # noqa: E501
                else:
                    label = value
            else:
                label = value

            if smry or (not analyze_min and not analyze_max):
                xvalue = [*range(0, len(yvalue))]
                axes[i].step(xvalue, yvalue, ".:", where="mid", label=label)
                axes[i].set_ylabel(yunit)
            else:
                if analyze_min:
                    yvalue = analyze[analyze_min]
                    xvalue = [*range(0, len(yvalue))]
                    label = f"{value}: {analyze_min}"
                    axes[i].step(
                        xvalue, yvalue, ".:", where="mid", label=label
                    )  # noqa: E501
                if analyze_max:
                    yvalue = analyze[analyze_max]
                    xvalue = [*range(0, len(yvalue))]
                    label = f"{value}: {analyze_max}"
                    axes[i].step(
                        xvalue, yvalue, ".:", where="mid", label=label
                    )  # noqa: E501
                axes[i].set_ylabel(aunit)
        axes[i].xaxis.set_major_locator(plot.MaxNLocator(integer=True))
        axes[i].set_title(entry.upper())
        axes[i].legend(loc="center left", fontsize="x-small")
        i = i + 1
    axes[i - 1].set_xlabel("Number of Builds")
    figure.suptitle("Performance History", fontsize="x-large")
    figure.gca().invert_xaxis()
    figure.tight_layout()
    # determine filename (graphics)
    figtypes = plot.gcf().canvas.get_supported_filetypes()
    argfig = pathlib.Path(args.figure)
    deffig = pathlib.Path(argd.figure)
    if argfig.is_dir():
        figloc = argfig
        figext = deffig.suffix
        figstm = deffig.stem
    elif argfig.suffix[1:] in figtypes.keys():
        figloc = argfig.parent
        figext = argfig.suffix
        figstm = argfig.stem
    elif "." == str(argfig.parent):
        figloc = argfig.parent
        figext = (
            f".{argfig.name}"
            if argfig.name in figtypes.keys()
            else deffig.suffix
        )
        figstm = deffig.stem
    else:
        figloc = argfig.parent
        figext = deffig.suffix
        figstm = argfig.stem if argfig.stem else deffig.stem
    if 0 < len(match):
        punct = str.maketrans("", "", "!\"#$%&'()*+-./:<=>?@[\\]^_`{|}~")
        clean = "-".join(
            [re.sub(r"[ ,;]+", "_", s.translate(punct)) for s in match]
        )
        figstm = f"{figstm}-{clean.lower()}"
    figout = figloc / f"{figstm}{figext}"
    # save graphics file
    figure.savefig(figout)


if __name__ == "__main__":
    path = pathlib.Path(__file__)
    here = path.absolute().parent
    base = path.stem
    argparser = argparse.ArgumentParser(
        description="Report results from Continuous Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "-f",
        "--filepath",
        type=pathlib.Path,
        default=here / f"{base}.json",
        help="JSON-database used to cache results",
    )
    argparser.add_argument(
        "-g",
        "--figure",
        type=str,
        default=f"{base}.png",
        help="Graphics format, filename, or path",
    )
    argparser.add_argument(
        "-i",
        "--infile",
        type=pathlib.Path,
        default=None,
        help="Input as if loaded from Buildkite",
    )
    argparser.add_argument(
        "-c",
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
        "-t",
        "--token",
        type=str,
        help="Authorization token",
    )
    argparser.add_argument(
        "-e",
        "--select",
        type=str,
        default="",
        help="Select entry",
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
        help="Plotted values",
    )
    argparser.add_argument(
        "-s",
        "--summary",
        type=str,
        default="gflops",
        help='If "", plot per-layer history',
    )
    argparser.add_argument(
        "-a",
        "--analyze",
        type=str,
        default="layer",
        help="Analyze common property",
    )
    argparser.add_argument(
        "-m",
        "--mean",
        type=int,
        default=3,
        help="Number of samples",
    )
    argparser.add_argument(
        "-n",
        "--history",
        type=int,
        default=30,
        help="Number of builds",
    )
    args = argparser.parse_args()
    argd = argparser.parse_args([])
    main(args, argd)
