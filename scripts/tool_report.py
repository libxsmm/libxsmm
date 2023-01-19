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
import datetime
import pathlib
import pickle
import json
import PIL
import sys
import re


def parselog(database, strbuild, jobname, txt, nentries, nerrors, select=None):
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
            if line
            and line.group(1)
            and all(32 <= ord(c) for c in line.group(1))
        ]
        if values and not any("syntax error" in v for v in values):
            category = match.group(1) if not select else select
            if strbuild not in database:
                database[strbuild] = dict()
            if category not in database[strbuild]:
                database[strbuild][category] = dict()
            if jobname not in database[strbuild][category]:
                database[strbuild][category][jobname] = dict()
            oldval = database[strbuild][category][jobname]
            if values != oldval:
                database[strbuild][category][jobname] = values
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


def matchstr(s1, s2, exact=False):
    if s1:
        if exact or (re.search(r"\d+$", s1) and re.search(r"\d+$", s2)):
            # avoid matching, e.g. "a12" if "a1" is searched
            return (s1 + ".") in (s2 + ".")
        else:
            return s1 in s2
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


def divup(a, b):
    return int((a + b - 1) / b)


def mtime(filename):
    try:
        return pathlib.Path(filename).stat().st_mtime if filename else 0
    except:  # noqa: E722
        return 0


def savedb(filename, database):
    if ".json" == filename.suffix:
        with open(filename, "w") as file:
            json.dump(database, file, indent=2)
            file.write("\n")  # append newline at EOF
    else:  # pickle
        with open(filename, "wb") as file:
            pickle.dump(database, file)


def main(args, argd):
    urlbase = "https://api.buildkite.com/v2/organizations"
    url = f"{urlbase}/{args.organization}/pipelines/{args.pipeline}/builds"
    auth = {"Authorization": f"Bearer {args.token}"} if args.token else None
    params = {"per_page": 100, "page": 1}
    if args.select:
        select = (
            args.select.lower().split()
            if not args.select_exact
            else [args.select.lower()]
        )
    else:
        select = []
    query = (
        args.query.lower().split()
        if not args.query_exact
        else [args.query.lower()]
        if args.query
        else []
    )
    smry = args.summary.lower()
    rslt = args.result.lower()
    sdo = 0 < args.mean and smry != rslt
    inflight = max(args.inflight, 0)
    nerrors = nentries = 0
    outfile = None
    match = []

    if args.infile and args.infile.is_file():
        try:
            with open(args.infile, "r") as file:
                txt = file.read()
        except:  # noqa: E722
            args.infile = None
            pass
        outfile = (
            pathlib.Path(f"{args.infile.stem}{argd.filepath.suffix}")
            if args.filepath == argd.filepath
            else args.filepath
        )
    elif args.infile is None:  # connect to URL
        outfile = args.filepath

    # timestamp before loading database
    ofmtime = mtime(outfile)
    if args.filepath.is_file():
        try:
            if ".json" == args.filepath.suffix:
                with open(args.filepath, "r") as file:
                    database = json.load(file)
            else:  # pickle
                with open(args.filepath, "rb") as file:
                    database = pickle.load(file)
        except Exception as error:
            msg = str(error).replace(": ", f" in {args.filepath.name}: ")
            print(f"ERROR: {msg}", file=sys.stderr)
            exit(1)
    else:
        database = dict()
    dbkeys = list(database.keys())
    latest = int(dbkeys[-1]) if dbkeys else 0

    if args.infile and args.infile.is_file():
        next = latest + 1
        nbld = (
            args.nbuild
            if (args.nbuild and 0 < args.nbuild and args.nbuild < next)
            else next
        )
        name = (
            args.query
            if args.query and (args.query != argd.query or args.query_exact)
            else args.infile.stem
        )
        nentries, nerrors = parselog(
            database,
            str(nbld),
            name,
            txt,
            nentries,
            nerrors,
            select=args.select,
        )
        if 0 < nentries:
            latest = next
    elif args.infile is None:  # connect to URL
        try:  # proceeed with cached results in case of an error
            builds = requests.get(url, params=params, headers=auth).json()
        except:  # noqa: E722
            builds = None
            pass
        if builds and "message" in builds:
            message = builds["message"]
            print(f"ERROR: {message}", file=sys.stderr)
            exit(1)
        elif not args.token:
            print("ERROR: token is missing!", file=sys.stderr)
            exit(1)
        elif not builds:
            print(f"WARNING: failed to connect to {url}.", file=sys.stderr)

        njobs = 0
        while builds:
            # iterate over all builds (latest first)
            for build in builds:
                nbuild = build["number"]
                strbuild = str(nbuild)  # JSON stores integers as string
                if (  # consider early exit
                    nbuild <= max(latest - inflight, 1)
                    and "running" != build["state"]
                    and strbuild in database
                ):
                    latest = nbuild
                    builds = None
                    break
                jobs = build["jobs"]
                n = 0
                for job in (job for job in jobs if 0 == job["exit_status"]):
                    if 2 <= args.verbosity or 0 > args.verbosity:
                        if 0 == n:
                            print(f"[{nbuild}]", end="", flush=True)
                        print(".", end="", flush=True)
                    log = requests.get(job["log_url"], headers=auth)
                    txt = json.loads(log.text)["content"]
                    nentries, nerrors = parselog(
                        database, strbuild, job["name"], txt, nentries, nerrors
                    )
                    n = n + 1
                if 0 == n and nbuild <= latest and "running" != build["state"]:
                    latest = nbuild
                    builds = None
                    break
                njobs = njobs + n
            if builds and 1 < nbuild:
                params["page"] = params["page"] + 1  # next page
                builds = requests.get(url, params=params, headers=auth).json()
            else:
                builds = None
        if 0 < njobs and (2 <= args.verbosity or 0 > args.verbosity):
            print("[OK]")

    # conclude loading data from latest CI
    if 2 <= args.verbosity or 0 > args.verbosity:
        if 0 != nerrors:
            y = "ies" if 1 != nerrors else "y"
            print(
                f"WARNING: ignored {nerrors} erroneous entr{y}!",
                file=sys.stderr,
            )
        y = "ies" if 1 != nentries else "y"
        print(f"Found {nentries} new entr{y}.")

    # save database (consider retention), and update dbkeys
    dbkeys = list(database.keys())
    dbsize = len(dbkeys)
    if 0 != nentries and ofmtime == mtime(outfile):
        if not outfile.exists() and (
            2 <= args.verbosity or 0 > args.verbosity
        ):
            print(f"{outfile} database created.")
        # sort by top-level key if database is to be stored (build number)
        database = dict(sorted(database.items(), key=lambda v: int(v[0])))
        # backup database and prune according to retention
        retention = max(args.retention, args.history)
        if 0 < retention and (2 * retention) < dbsize:
            nowutc = datetime.datetime.now(datetime.timezone.utc)
            nowstr = nowutc.strftime("%Y%m%d")  # day
            newname = f"{outfile.stem}-{nowstr}{outfile.suffix}"
            retfile = outfile.parent / newname
            if not retfile.exists():
                savedb(retfile, database)  # unpruned
                for key in dbkeys[0 : dbsize - retention]:  # noqa: E203
                    del database[key]
                dbkeys = list(database.keys())
                dbsize = retention
        savedb(outfile, database)

    # collect categories for template (figure)
    templidx = (
        1  # file-based input (just added) shall determine template
        if (args.infile and args.infile.is_file())
        else min(inflight + 1, dbsize)
    )
    templkey = dbkeys[-templidx] if dbkeys else ""  # string
    template = database[templkey] if templkey in database else []
    entries = [
        e  # category (one level below build number)
        for e in template
        if not select
        or any(matchstr(s, e.lower(), exact=args.select_exact) for s in select)
    ]
    if entries and not select and args.select_exact:
        entries = [entries[-1]]  # assume insertion order is preserved

    # determine image resolution
    rdef = [int(r) for r in argd.resolution.split("x")]
    if 2 == len(rdef):
        rdef.append(100)
    rstr = args.resolution.split("x")
    rint = []
    for i in range(len(rdef)):
        try:
            rint.append(int(rstr[i]))
        except:  # noqa: E722
            rint.append(
                rdef[i] if 1 != i else round(rint[0] * rdef[1] / rdef[0])
            )

    # setup figure
    figure, axes = plot.subplots(
        max(len(entries), 1),
        sharex=True,
        figsize=(divup(rint[0], rint[2]), divup(rint[1], rint[2])),
        dpi=rint[2],
    )
    if 2 > len(entries):
        axes = [axes]

    # build figure
    i = 0
    infneg = float("-inf")
    infpos = float("inf")
    nvalues = 0
    yunit = None
    for entry in entries:
        n = 0
        for value in (
            v
            for v in template[entry]
            if not query or any(matchstr(p, v.lower()) for p in query)
        ):
            yvalue = []  # determined by --result
            meanvl = []  # determined by --summary
            sunit = aunit = None
            analyze = dict()
            analyze_min = ""
            analyze_max = ""
            if value not in match:
                match.append(value)
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
                            vmax = vmin = 0
                            amax = infneg
                            amin = infpos
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
                            if analyze_min and 0 != vmin and infpos != amin:
                                vlabel = analyze_min.replace(" ", "")
                                label = f"{label} {vlabel}={vmin}{unit} ({num2str(amin)}%)"  # noqa: E501
                            if analyze_max and 0 != vmax and infneg != amax:
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
            n = n + 1
        nvalues = max(nvalues, n)
        axes[i].xaxis.set_major_locator(plot.MaxNLocator(integer=True))
        axes[i].set_title(entry.upper())
        axes[i].legend(loc="center left", fontsize="x-small")
        i = i + 1
    axes[i - 1].set_xlabel("Number of Builds")
    figure.suptitle("Performance History", fontsize="x-large")
    figure.gca().invert_xaxis()
    figure.tight_layout()

    # determine supported file types and filename components
    figcanvas = figure.canvas
    figtypes = figcanvas.get_supported_filetypes()
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

    # determine filename from components
    punct = str.maketrans("", "", "!\"#$%&'()*+-./:<=>?@[\\]^_`{|}~")
    figcat = (
        ""
        if 1 < len(entries) or 0 == len(entries)
        else f"-{entries[0].translate(punct)}"
    )
    if 0 < len(match):
        clean = [re.sub(r"[ ,;]+", "_", s.translate(punct)) for s in match]
        parts = [s.lower() for c in clean for s in c.split("_")]
        fixqry = f"-{'_'.join(dict.fromkeys(parts))}"
    else:
        fixqry = ""
    figout = figloc / f"{figstm}{fixqry}{figcat}{figext}"

    # reduce file size (png) and save figure
    if ".png" == figout.suffix:
        figcanvas.draw()  # otherwise the image is empty
        image = PIL.Image.frombytes("RGB", rint[0:2], figcanvas.tostring_rgb())
        ncolors = divup(nvalues + 2, 16) * 16
        palette = PIL.Image.Palette.ADAPTIVE
        image = image.convert("P", palette=palette, colors=ncolors)
        image.save(figout, "PNG", optimize=True)
    else:
        figure.savefig(figout)  # save graphics file
    if 1 == args.verbosity or 0 > args.verbosity:
        print(f"{figout} created.")


if __name__ == "__main__":
    path = pathlib.Path(__file__)
    here = path.absolute().parent
    base = path.stem
    argparser = argparse.ArgumentParser(
        description="Report results from Continuous Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="0: quiet, 1: automation, 2: progress",
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
        "-d",
        "--resolution",
        type=str,
        default="900x600",
        help="Graphics WxH[xDPI]",
    )
    argparser.add_argument(
        "-i",
        "--infile",
        type=pathlib.Path,
        default=None,
        help="Input data as if loaded from Buildkite",
    )
    argparser.add_argument(
        "-j",
        "--nbuild",
        type=int,
        default=None,
        help="Where to insert, not limited to infile",
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
        "-x",
        "--query-exact",
        action="store_true",
        help="Exact query",
    )
    argparser.add_argument(
        "-y",
        "--query",
        type=str,
        default="resnet",
        help="Set of values",
    )
    argparser.add_argument(
        "-z",
        "--select-exact",
        action="store_true",
        help="Exact select",
    )
    argparser.add_argument(
        "-s",
        "--select",
        type=str,
        default=None,
        help="Category, all if none",
    )
    argparser.add_argument(
        "-r",
        "--result",
        type=str,
        default="ms",
        help="Plotted values",
    )
    argparser.add_argument(
        "-a",
        "--analyze",
        type=str,
        default=None,
        help='Common property, e.g., "layer"',
    )
    argparser.add_argument(
        "-b",
        "--summary",
        type=str,
        default="gflops",
        help='If "", plot per-layer history',
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
    argparser.add_argument(
        "-u",
        "--retention",
        type=int,
        default=60,
        help="Keep history",
    )
    argparser.add_argument(
        "-k",
        "--inflight",
        type=int,
        default=2,
        help="Re-scan builds",
    )
    args = argparser.parse_args()
    argd = argparser.parse_args([])
    main(args, argd)
