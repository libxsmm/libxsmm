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
import tempfile
import pathlib
import pickle
import json
import PIL
import sys
import re
import os


def depth(obj):
    result = 0
    if not isinstance(obj, str):
        try:  # iterable?
            for i in obj:
                item = obj[i] if isinstance(obj, dict) else i
                result = max(result, depth(item) + 1)
        except:  # noqa: E722
            pass
    return result


def matchstr(s1, s2, exact=False):
    if s1:
        if exact or (re.search(r"\d+$", s1) and re.search(r"\d+$", s2)):
            # avoid matching, e.g. "a12" if "a1" is searched
            return (s1 + ".") in (s2 + ".")
        else:
            return s1 in s2
    else:
        return False


def matchlst(string, strlst, exact=False):
    for s in strlst:
        if matchstr(string, s.lower(), exact):
            return s
    return ""


def parsename(string):
    parts = string.split()
    result = parts[0]
    for name in parts:
        if re.match(r"[a-zA-Z]+(?:[0-9_\-]+[a-zA-Z]+)*$", name):
            result = name
    return result


def parseval(string):
    """
    Split "string" into "init", "value", and "unit".
    """
    return re.match(
        r"(.+)?(^|[\s:=])([+\-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+\-]?\d+)?)",
        string,  # noqa: E501
    )


def parselog(database, strbuild, jobname, txt, nentries, nerrors):
    pattern = (
        r"^\+\+\+ PERFORMANCE ([a-zA-Z]+(?:[0-9_\-,]+[a-zA-Z]+)*)([^\+\-]+)"
    )
    matches = [
        match
        for match in re.finditer(pattern, txt, re.MULTILINE | re.DOTALL)
        if match and match.group(1) and match.group(2)
    ]
    if matches:  # attempt to match native format (telegram)
        invalid = ["syntax error", "ERROR:", "Traceback", '\\"']
        for match in matches:
            values = [
                line.group(1)
                for line in re.finditer(r"([^\n\r]+)", match.group(2))
                if (line and line.group(1) and 4 >= len(line.group(1).split()))
                and all(i not in line.group(1) for i in invalid)
                and all(32 <= ord(c) for c in line.group(1))
            ]
            if values:
                category = match.group(1).replace(",", " ")
                if strbuild not in database:
                    database[strbuild] = dict()
                if category not in database[strbuild]:
                    database[strbuild][category] = dict()
                if jobname not in database[strbuild][category]:
                    nentries = nentries + 1
                else:
                    nerrors = nerrors + 1
                database[strbuild][category][jobname] = values
            else:
                nerrors = nerrors + 1
    else:  # attempt to match inlined JSON section
        pattern = r"--partition=([a-zA-Z]+(?:[0-9_\-,]+[a-zA-Z]+)*).+(^{.+})"
        matches = [
            match
            for match in re.finditer(pattern, txt, re.MULTILINE | re.DOTALL)
            if match and match.group(1) and match.group(2)
        ]
        if not matches:
            pattern = (  # JSON-only (not a full logfile)
                r"^\+\+\+ REPORT ([a-zA-Z]+(?:[0-9_\-,]+[a-zA-Z]+)*).+(^{.+})"
            )
            matches = [
                match
                for match in re.finditer(
                    pattern, txt, re.MULTILINE | re.DOTALL
                )
                if match and match.group(1) and match.group(2)
            ]
        for match in matches:
            try:
                clean = (  # fixup somewhat malformed JSON
                    match.group(2)
                    .replace("'", '"')
                    .replace("True", "true")
                    .replace("False", "false")
                )
                values = json.loads(clean)
            except:  # noqa: E722
                values = None
            if values:
                category = match.group(1).replace(",", " ")
                if strbuild not in database:
                    database[strbuild] = dict()
                if category not in database[strbuild]:
                    database[strbuild][category] = dict()
                if 1 < depth(values):
                    for i in values:
                        if i not in database[strbuild][category]:
                            nentries = nentries + 1
                        else:
                            nerrors = nerrors + 1
                        database[strbuild][category][i] = values[i]
                else:
                    if jobname not in database[strbuild][category]:
                        nentries = nentries + 1
                    else:
                        nerrors = nerrors + 1
                    database[strbuild][category][jobname] = values
            else:
                nerrors = nerrors + 1
    return nentries, nerrors


def mtime(filename):
    try:
        os.sync()  # flush pending buffers
        return pathlib.Path(filename).stat().st_mtime if filename else 0
    except:  # noqa: E722
        return 0


def savedb(filename, database, filetime=None):
    if not filename.is_dir():
        tmpfile = tempfile.mkstemp(
            filename.suffix, filename.stem + ".", filename.parent
        )
        if ".json" == filename.suffix.lower():
            with os.fdopen(tmpfile[0], "w") as file:
                json.dump(database, file, indent=2)
                file.write("\n")  # append newline at EOF
        else:  # pickle
            with os.fdopen(tmpfile[0], "wb") as file:
                pickle.dump(database, file)
        # os.close(tmpfile[0])
        if not filetime or filetime == mtime(filename):
            if filename.exists():
                os.replace(tmpfile[1], filename)
            else:
                os.rename(tmpfile[1], filename)
    else:
        print("WARNING: no database created or updated.", file=sys.stderr)


def num2fix(num, decimals=0):
    dec = pow(10, decimals)
    return int((dec * num + 0.5) if 0 <= num else (dec * num - 0.5)) / dec


def num2str(num):
    return (
        f"$\pm${num}"  # noqa: W605
        if 0 == num
        else (f"+{num}" if 0 < num else f"{num}")
    )


def divup(a, b):
    return int((a + b - 1) / b)


def main(args, argd):
    urlbase = "https://api.buildkite.com/v2/organizations"
    url = (
        f"{urlbase}/{args.organization}/pipelines/{args.pipeline}/builds"
        if args.pipeline
        else ""
    )
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
    accuracy = 1
    match = []

    if args.infile and (args.infile.is_file() or args.infile.is_fifo()):
        try:
            with open(args.infile, "r") as file:
                txt = file.read()
        except:  # noqa: E722
            args.infile = None
            pass
        outfile = (
            pathlib.Path(f"{args.infile.stem}{argd.filepath.suffix}")
            if args.filepath == argd.filepath
            or not (args.filepath.is_file() or args.filepath.is_fifo())
            else args.filepath
        )
    elif args.infile is None:  # connect to URL
        outfile = args.filepath

    # timestamp before loading database
    ofmtime = mtime(outfile)
    if args.filepath.is_file() or args.filepath.is_fifo():
        try:
            if ".json" == args.filepath.suffix.lower():
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

    # attempt to load weights
    wfile = (
        args.weights
        if (args.weights.is_file() or args.weights.is_fifo())
        else argd.weights
    )
    if wfile.is_file() or wfile.is_fifo():
        try:
            if ".json" == wfile.suffix.lower():
                with open(wfile, "r") as file:
                    weights = json.load(file)
            else:  # pickle
                with open(wfile, "rb") as file:
                    weights = pickle.load(file)
        except Exception as error:
            msg = str(error).replace(": ", f" in {wfile.name}: ")
            print(f"ERROR: {msg}", file=sys.stderr)
            exit(1)
    else:
        weights = {}

    # populate default weights
    write = None
    for build in database.values():
        for entries in build.values():
            for key, entry in entries.items():
                name = parsename(key)
                if name not in weights:
                    write = [1.0 for e in entry if ":" in e]
                    if write:
                        weights[name] = write
    if write:  # write weights if modified (not wfile)
        savedb(args.weights, weights)

    if args.infile and (args.infile.is_file() or args.infile.is_fifo()):
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
            database, str(nbld), name, txt, nentries, nerrors
        )
        if 0 < nentries:
            latest = next
    elif args.infile is None and url:  # connect to URL
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

        nbuilds = njobs = 0
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
                for job in (
                    job
                    for job in jobs
                    if "exit_status" in job and 0 == job["exit_status"]
                ):
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
                nbuilds = nbuilds + 1
                njobs = njobs + n
                if (
                    0 == n and nbuild <= latest and "running" != build["state"]
                ) or args.history <= nbuilds:
                    latest = nbuild
                    builds = None
                    break
            if builds and 1 < nbuild:
                params["page"] = params["page"] + 1  # next page
                builds = requests.get(url, params=params, headers=auth).json()
            else:
                builds = None
        if (2 <= args.verbosity or 0 > args.verbosity) and 0 < njobs:
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
    if 0 != nentries:
        # sort by top-level key if database is to be stored (build number)
        database = dict(sorted(database.items(), key=lambda v: int(v[0])))
        # backup database and prune according to retention
        retention = max(args.retention, args.history)
        if 0 < retention and (retention + args.history) < dbsize:
            nowutc = datetime.datetime.now(datetime.timezone.utc)
            nowstr = nowutc.strftime("%Y%m%d")  # day
            retfile = outfile.with_name(
                f"{outfile.stem}.{nowstr}{outfile.suffix}"
            )
            if not retfile.exists():
                savedb(retfile, database)  # unpruned
                for key in dbkeys[0 : dbsize - retention]:  # noqa: E203
                    del database[key]
                dbkeys = list(database.keys())
                dbsize = retention
        savedb(outfile, database, ofmtime)
        if (  # print filename of database
            2 <= args.verbosity or 0 > args.verbosity
        ) and not outfile.exists():
            print(f"{outfile} database created.")

    if dbkeys:  # collect categories for template (figure)
        if args.nbuild in dbkeys:
            templkey = dbkeys[args.nbuild]
        elif not args.infile or not (
            args.infile.is_file() or args.infile.is_fifo()
        ):
            templkey = dbkeys[-min(inflight + 1, dbsize)]
        else:  # file-based input (just added)
            templkey = dbkeys[-1]
        template = database[templkey]
    else:
        template = dict()

    entries = [
        e  # category (one level below build number)
        for e in template
        if not select
        or any(matchstr(s, e.lower(), args.select_exact) for s in select)
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
            r = rdef[i] if 1 != i else round(rint[0] * rdef[1] / rdef[0])
            rint.append(r)

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
    ngraphs = i = 0
    infneg = float("-inf")
    infpos = float("inf")
    yunit = None
    for entry in entries:
        n = 0
        for value in (
            v
            for v in template[entry]
            if not query
            or eval(args.query_op)(matchstr(p, v.lower()) for p in query)
        ):
            xvalue = []  # build numbers corresponding to yvalue
            yvalue = []  # determined by --result
            meanvl = []  # determined by --summary
            sunit = aunit = None
            layers = dict()
            layers_min = ""
            layers_max = ""
            if value not in match:
                match.append(value)
            # collect data to be plotted
            for build in (
                b
                for b in database
                if entry in database[b] and value in database[b][entry]
            ):
                ylabel = slabel = None
                values = database[build][entry][value]
                if isinstance(values, dict):
                    qry = rslt.split(",")
                    key = matchlst(qry[0], values.keys())
                    if key:
                        scale = 1.0 if 2 > len(qry) else float(qry[1])
                        parsed = parseval(values[key])
                        unit = values[key][
                            parsed.end(3) :  # noqa: E203
                        ].strip()
                        yvalue.append(float(values[key].split()[0]) * scale)
                        xvalue.append(build)  # string
                        ylabel = (
                            (unit if unit else key) if 3 > len(qry) else qry[2]
                        )
                    if sdo:
                        qry = smry.split(",")
                        key = matchlst(qry[0], values.keys())
                        if key:
                            scale = 1.0 if 2 > len(qry) else float(qry[1])
                            parsed = parseval(values[key])
                            unit = values[key][
                                parsed.end(3) :  # noqa: E203
                            ].strip()
                            meanvl.append(
                                float(values[key].split()[0]) * scale
                            )
                            slabel = unit if unit else key
                    if not slabel:
                        slabel = ylabel
                else:
                    # match --result primarily against "unit"
                    for v in reversed(values):  # match last entry
                        parsed = parseval(v)
                        if parsed and parsed.group(3):
                            unit = v[parsed.end(3) :].strip()  # noqa: E203
                            ulow = unit.lower()
                            if not ylabel and matchstr(rslt, ulow):
                                yvalue.append(float(parsed.group(3)))
                                xvalue.append(build)  # string
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
                                xvalue.append(build)  # string
                                ylabel = ulab
                            if not slabel and sdo and matchstr(smry, ilow):
                                meanvl.append(float(parsed.group(3)))
                                slabel = ulab
                            if (init and (not aunit or ulab == aunit)) and (
                                not args.analyze
                                or matchstr(args.analyze, ilow)
                            ):
                                if init not in layers:
                                    if not aunit:
                                        aunit = ulab
                                    layers[init] = []
                                layers[init].append(float(parsed.group(3)))

            j = 0
            s = args.history
            wname = value.split()[0]
            wlist = weights[wname] if wname in weights else []
            wdflt = True  # only default-weights discovered
            # (re-)reverse, trim, and apply weights
            for a in reversed(layers):
                y = layers[a]
                s = min(s, len(y))
                w = wlist[j] if j < len(wlist) else 1.0
                if 1.0 != w:
                    layers[a] = [y[len(y) - k - 1] * w for k in range(s)]
                    if wdflt:
                        wdflt = False
                else:  # unit-weight
                    layers[a] = [y[len(y) - k - 1] for k in range(s)]
                j = j + 1
            if not yunit:
                yunit = (ylabel if ylabel else args.result).split()[0]
            # summarize layer into yvalue only in case of non-default weights
            if (not aunit or aunit == yunit) and not wdflt:
                yvalue = [sum(y) for y in zip(*layers.values())]
            elif yvalue:  # (re-)reverse and trim
                yvalue = yvalue[: -args.history - 1 : -1]  # noqa: E203
            if xvalue and yvalue:  # (re-)reverse and trim
                xvalue = xvalue[: -len(yvalue) - 1 : -1]  # noqa: E203

            # collect statistics and perform some analysis
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
                    label = f"{value} = {num2fix(mnew, accuracy)} {sunit}"
                    if vold:
                        mold = statistics.geometric_mean(vold)
                        perc = num2fix(100 * (mnew - mold) / mold)
                        label = f"{label} ({num2str(perc)}%)"

                        if 0 != perc and args.analyze:
                            vmax = vmin = 0
                            amax = infneg
                            amin = infpos
                            for a in layers:
                                values = [v for v in layers[a] if 0 < v]
                                vnew = values[0 : args.mean]  # noqa: E203
                                vold = values[args.mean :]  # noqa: E203
                                if vnew and vold:
                                    anew = statistics.geometric_mean(vnew)
                                    aold = statistics.geometric_mean(vold)
                                    perc = num2fix(100 * (anew - aold) / aold)
                                    if perc > amax:
                                        vmax = num2fix(anew, accuracy)
                                        layers_max = a
                                        amax = perc
                                    elif perc < amin:
                                        vmin = num2fix(anew, accuracy)
                                        layers_min = a
                                        amin = perc
                            unit = f" {aunit}" if aunit else ""
                            if layers_min and 0 != vmin and infpos != amin:
                                vlabel = layers_min.replace(" ", "")
                                label = f"{label} {vlabel}={vmin}{unit} ({num2str(amin)}%)"  # noqa: E501
                            else:
                                layers_min = ""
                            if layers_max and 0 != vmax and infneg != amax:
                                vlabel = layers_max.replace(" ", "")
                                label = f"{label} {vlabel}={vmax}{unit} ({num2str(amax)}%)"  # noqa: E501
                            else:
                                layers_max = ""
                else:
                    label = value
            else:
                label = value

            # determine size of shared x-axis
            xsize = args.history
            if smry and (not aunit or aunit == yunit):
                xsize = min(len(yvalue), xsize)
            for a in layers:
                if a == layers_min or not smry:
                    xsize = min(len(layers[a]), xsize)
                if a == layers_max and smry:
                    xsize = min(len(layers[a]), xsize)
            xrange = range(xsize)

            # plot values and legend as collected above
            if smry and (not aunit or aunit == yunit):
                axes[i].step(
                    xrange, yvalue[0:xsize], ".:", where="mid", label=label
                )
                axes[i].set_ylabel(yunit)
                n = n + 1
            for a in layers:
                if a == layers_min or not smry:
                    yvalue = layers[a][0:xsize]
                    label = f"{value}: {a}"
                    axes[i].step(
                        xrange, yvalue, ".:", where="mid", label=label
                    )  # noqa: E501
                    axes[i].set_ylabel(aunit)
                    n = n + 1
                if a == layers_max and smry:
                    yvalue = layers[a][0:xsize]
                    label = f"{value}: {a}"
                    axes[i].step(
                        xrange, yvalue, ".:", where="mid", label=label
                    )  # noqa: E501
                    axes[i].set_ylabel(aunit)
                    n = n + 1
            axes[i].xaxis.set_ticks(xrange)  # before set_xticklabels
            axes[i].set_xticklabels(xvalue[0:xsize])
        ngraphs = max(ngraphs, n)
        if 0 < ngraphs:
            axes[i].xaxis.set_major_locator(plot.MaxNLocator(integer=True))
            axes[i].set_title(entry.upper())
            axes[i].legend(loc="center left", fontsize="x-small")
        i = i + 1
    axes[i - 1].set_xlabel("Build Number")
    figure.suptitle("Performance History", fontsize="x-large")
    figure.gca().invert_xaxis()
    figure.tight_layout()

    if 0 < ngraphs:
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
        figcat = re.sub(
            r"[ ,;]+",
            "_",
            ""
            if 1 < len(entries) or 0 == len(entries)
            else f"-{entries[0].translate(punct)}",
        )
        if 0 < len(match):
            clean = [re.sub(r"[ ,;]+", "_", s.translate(punct)) for s in match]
            parts = [s.lower() for c in clean for s in c.split("_")]
            fixqry = f"-{'_'.join(dict.fromkeys(parts))}"
        else:
            fixqry = ""
        figout = figloc / f"{figstm}{fixqry}{figcat}{figext}"

        # reduce file size (png) and save figure
        if ".png" == figout.suffix.lower():
            figcanvas.draw()  # otherwise the image is empty
            imageraw = figcanvas.tostring_rgb()
            image = PIL.Image.frombytes("RGB", rint[0:2], imageraw)
            # avoid Palette.ADAPTIVE, consider back/foreground color
            image = image.convert("P", colors=ngraphs + 2)
            image.save(figout, "PNG", optimize=True)
        else:
            figure.savefig(figout)  # save graphics file
        if 1 == args.verbosity or 0 > args.verbosity:
            print(f"{figout} created.")


if __name__ == "__main__":
    path = pathlib.Path(__file__)
    here = path.absolute().parent
    try:
        rdir = here.relative_to(pathlib.Path.cwd())
    except ValueError:
        rdir = here
    base = path.stem
    figtype = "png"

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
        "-w",
        "--weights",
        type=pathlib.Path,
        default=rdir / f"{base}.weights.json",
        help="Database to load weights",
    )
    argparser.add_argument(
        "-f",
        "--filepath",
        type=pathlib.Path,
        default=rdir / f"{base}.json",
        help="Database to store results",
    )
    argparser.add_argument(
        "-g",
        "--figure",
        type=str,
        default=f"{base}.{figtype}",
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
        "-u",
        "--query-op",
        type=str,
        default="all",
        choices=["all", "any"],
        help="Inexact query operator",
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
        default="resnet-50",
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
        default="ms",
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
        "-k",
        "--retention",
        type=int,
        default=60,
        help="Keep history",
    )
    argparser.add_argument(
        "-e",
        "--inflight",
        type=int,
        default=2,
        help="Re-scan builds",
    )

    args = argparser.parse_args()  # 1st pass
    if args.pipeline:
        filepath = rdir / f"{args.pipeline}.json"
        figure = f"{args.pipeline}.{figtype}"
        argparser.set_defaults(filepath=filepath, figure=figure)
        args = argparser.parse_args()  # 2nd pass
    if args.filepath.name:
        weights = args.filepath.with_name(
            f"{args.filepath.stem}.weights{args.filepath.suffix}"
        )
        argparser.set_defaults(weights=weights)
        args = argparser.parse_args()  # 3rd pass
    argd = argparser.parse_args([])

    main(args, argd)
