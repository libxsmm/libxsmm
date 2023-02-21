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
import statistics  # noqa: F401
import requests
import argparse
import datetime
import tempfile
import pathlib
import pickle
import json
import stat
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
    return [s for s in strlst if matchstr(string, s.lower(), exact)]


def matchop(op, value, query, exact=False):
    if query:
        if "not" != op:
            if op:
                result = eval(op)(matchstr(q, value.lower()) for q in query)
            else:  # any
                result = any(matchstr(q, value.lower()) for q in query)
        else:  # not
            result = all(not matchstr(q, value.lower()) for q in query)
    else:
        result = True
    return result


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


def sortdb(database):
    try:  # treat top-level key as integer (build number)
        result = dict(sorted(database.items(), key=lambda v: int(v[0])))
    except ValueError:
        result = dict(sorted(database.items()))
    for key, value in result.items():
        if isinstance(value, dict):
            result[key] = sortdb(value)
        else:
            result[key] = value
    return result


def loaddb(filename):
    result = dict()
    if filename.is_file() or filename.is_fifo():
        try:
            if ".json" == filename.suffix.lower():
                with open(filename, "r") as file:
                    result = json.load(file)
            else:  # pickle
                with open(filename, "rb") as file:
                    result = pickle.load(file)
        except Exception as error:
            msg = str(error).replace(": ", f" in {filename.name}: ")
            print(f"ERROR: {msg}", file=sys.stderr)
            exit(1)
    return result


def savedb(filename, database, filetime=None, retry=None):
    if filename and not filename.is_dir():
        tmpfile = tempfile.mkstemp(  # create temporary file
            filename.suffix, filename.stem + ".", filename.parent
        )
        if filename.exists():  # adopt permissions
            mode = stat.S_IMODE(os.stat(filename).st_mode)
        else:  # determine permissions
            umask = os.umask(0o666)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.fchmod(tmpfile[0], mode)
        max_retry = retry if retry else 1
        for i in range(max_retry):
            database = sortdb(database)
            if ".json" == filename.suffix.lower():
                with os.fdopen(tmpfile[0], "w") as file:
                    json.dump(database, file, indent=2)
                    file.write("\n")  # append newline at EOF
            else:  # pickle
                with os.fdopen(tmpfile[0], "wb") as file:
                    pickle.dump(database, file)
            if filename.exists():
                now = mtime(filename) if filetime else 0
                if not filetime or filetime == now:
                    os.replace(tmpfile[1], filename)
                    break
                elif (i + 1) < max_retry:  # retry
                    filetime = now
                    updated = loaddb(filename)
                    database.update(updated)
                else:
                    os.unlink(tmpfile[1])
            else:
                os.rename(tmpfile[1], filename)
                break
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


def mean2label(meanfn, size, values, init, unit, accuracy):
    nonzero = [v for v in values if 0 < v]
    vnew = values[0:size]  # noqa: E203
    result = ""
    if vnew:
        mnew = eval(meanfn)(vnew)
        vold = nonzero[size:]  # noqa: E203
        result = f"{init} = {num2fix(mnew, accuracy)} {unit}"
        if vold:
            mold = eval(meanfn)(vold)
            perc = num2fix(100 * (mnew - mold) / mold)
            result = f"{result} ({num2str(perc)}%)"
    return result


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
    rslt = args.result.lower()
    inflight = max(args.inflight, 0)
    nerrors = nentries = 0
    outfile = None
    accuracy = 1
    match = []

    if args.infile and (args.infile.is_file() or args.infile.is_fifo()):
        try:
            with open(args.infile, "r") as file:
                txt = file.read()
            if 0 > args.verbosity:
                print(txt)
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
    database = loaddb(args.filepath)
    dbkeys = list(database.keys())
    latest = int(dbkeys[-1]) if dbkeys else 0

    # attempt to load weights
    wfile = (
        args.weights
        if (args.weights.is_file() or args.weights.is_fifo())
        else argd.weights
    )
    wfmtime = mtime(wfile)
    weights = loaddb(wfile)

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
    if write:  # write weights if modified
        savedb(args.weights, weights, wfmtime, 2)

    nbuild = int(args.nbuild) if args.nbuild else 0
    if args.infile and (args.infile.is_file() or args.infile.is_fifo()):
        next = latest + 1
        nbld = nbuild if (0 < nbuild and nbuild < next) else next
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
                ibuild = build["number"]
                strbuild = str(ibuild)  # JSON stores integers as string
                if (  # consider early exit
                    ibuild <= max(latest - inflight, 1)
                    and "running" != build["state"]
                    and strbuild in database
                ):
                    latest = ibuild
                    builds = None
                    break
                jobs = build["jobs"]
                n = 0
                for job in (
                    job
                    for job in jobs
                    if "exit_status" in job and 0 == job["exit_status"]
                ):
                    if 2 <= abs(args.verbosity):
                        if 0 == n:
                            print(f"[{ibuild}]", end="", flush=True)
                        print(".", end="", flush=True)
                    log = requests.get(job["log_url"], headers=auth)
                    txt = json.loads(log.text)["content"]
                    nentries, nerrors = parselog(
                        database, strbuild, job["name"], txt, nentries, nerrors
                    )
                    n = n + 1
                nbuilds = nbuilds + 1
                njobs = njobs + n
                if (  # consider early exit
                    0 == n and ibuild <= latest and "running" != build["state"]
                ) or (args.history <= nbuilds or nbuild == ibuild):
                    latest = ibuild
                    builds = None
                    break
            if builds and 1 < ibuild:
                params["page"] = params["page"] + 1  # next page
                builds = requests.get(url, params=params, headers=auth).json()
            else:
                builds = None
        if 2 <= abs(args.verbosity) and 0 < njobs:
            print("[OK]")

    # conclude loading data from latest CI
    if 2 <= abs(args.verbosity):
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
        savedb(outfile, database, ofmtime, 2)
        if 2 <= abs(args.verbosity) and outfile and not outfile.exists():
            print(f"{outfile} database created.")

    if dbkeys:  # collect categories for template (figure)
        if nbuild in dbkeys:
            templkey = dbkeys[nbuild]
        elif not args.infile or not (
            args.infile.is_file() or args.infile.is_fifo()
        ):
            templkey = dbkeys[-min(inflight + 1, dbsize)]
        else:  # file-based input (just added)
            templkey = dbkeys[-1]
        template = database[templkey]
    else:
        template = dict()

    entries = [  # category (one level below build number)
        e for e in template if matchop("any", e, select, args.select_exact)
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
    query_op = args.query_op if args.query_op else argd.query_op
    transpat = "!\"#$%&'()*+-./:<=>?@[\\]^_`{|}~"
    split = str.maketrans(transpat, " " * len(transpat))
    clean = str.maketrans("", "", transpat)
    ngraphs = i = 0
    yunit = None
    addon = ""
    for entry in entries:
        n = 0
        for value in (
            v for v in template[entry] if matchop(query_op, v, query)
        ):
            layers, xvalue, yvalue = dict(), [], []
            legend, aunit = value, None
            if value not in match:
                match.append(value)
            # collect data to be plotted
            for build in (
                b
                for b in database
                if entry in database[b] and value in database[b][entry]
            ):
                ylabel = None
                values = database[build][entry][value]
                if isinstance(values, dict):  # JSON-format
                    qlst = rslt.split(",")
                    keys = matchlst(qlst[0], values.keys())
                    vals, legd = [], []
                    for key in keys:
                        scale = 1.0 if 2 > len(qlst) else float(qlst[1])
                        strval = str(values[key])  # ensure string
                        parsed = parseval(strval)
                        unit = strval[parsed.end(3) :].strip()  # noqa: E203
                        vals.append(float(strval.split()[0]) * scale)
                        if not ylabel:
                            ylabel = (
                                (unit if unit else key)
                                if 3 > len(qlst)
                                else qlst[2]
                            )
                        lst = key.translate(split).split()
                        detail = [s for s in lst if s.lower() != qlst[0]]
                        legd.append(
                            f"{value}_{'_'.join(detail)}" if detail else value
                        )
                    if vals:
                        if 1 < len(legd):
                            if not addon:
                                addon = rslt.split(",")[0].upper()
                            yvalue.append(vals)
                            legend = legd
                        else:
                            yvalue.append(vals[0])
                            legend = legd[0]
                        xvalue.append(build)  # string
                else:  # telegram format
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
                            if init and (not aunit or ulab == aunit):
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
            layerkeys = list(layers.keys())
            for a in reversed(layerkeys):
                y = layers[a]
                s = min(s, len(y))
                w = wlist[j] if j < len(wlist) else 1.0
                if 1.0 != w:
                    layers[a] = [y[len(y) - k - 1] * w for k in range(s)]
                    wdflt = False
                else:  # unit-weight
                    layers[a] = [y[len(y) - k - 1] for k in range(s)]
                j = j + 1
            if not yunit and (ylabel or args.result):
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
                fn = (
                    "statistics.geometric_mean"
                    if hasattr(statistics, "geometric_mean")
                    else "statistics.median"
                )
                if isinstance(legend, list):
                    ylist = list(zip(*yvalue))
                    label = []
                    for j in range(len(legend)):
                        y, z = ylist[j], legend[j]
                        s = mean2label(fn, args.mean, y, z, yunit, accuracy)
                        label.append(s)
                else:
                    label = mean2label(
                        fn, args.mean, yvalue, legend, yunit, accuracy
                    )
            else:
                label = legend

            # determine size of shared x-axis
            xsize = args.history
            if not aunit or aunit == yunit:
                xsize = min(len(yvalue), xsize)
            yvalue = yvalue[0:xsize]
            xrange = range(xsize)

            # plot values and legend as collected above
            if not aunit or aunit == yunit:
                axes[i].step(xrange, yvalue, ".:", where="mid", label=label)
                axes[i].set_ylabel(yunit)
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
    title = "Performance History"
    figure.suptitle(
        f"{title} ({addon})" if addon else title, fontsize="x-large"
    )
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
        figdet = (
            ""  # eventually add details about category
            if 1 < len(entries) or 0 == len(entries)
            else f"-{entries[0].translate(clean)}"
        )
        figcat = re.sub(r"[ ,;]+", "_", figdet)
        if 0 < len(match):
            match = [re.sub(r"[ ,;]+", "_", s.translate(clean)) for s in match]
            parts = [s.lower() for c in match for s in c.split("_")]
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
        if 1 == abs(args.verbosity):
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
        help="0: quiet, 1: automation, 2: progress, negative: echo input",
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
        type=str,
        default=None,
        help="Where to insert, not limited to infile",
    )
    argparser.add_argument(
        "-a",
        "--token",
        type=str,
        help="Authorization token",
    )
    argparser.add_argument(
        "-b",
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
        "-u",
        "--query-op",
        type=str,
        default="all",
        choices=["all", "any", "not", ""],
        help="Query operator",
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
        "-c",
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
