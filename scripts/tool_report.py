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
import requests
import argparse
import datetime
import tempfile
import pathlib
import pickle
import numpy
import math
import copy
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
            if exact:
                return s1 == s2
            else:  # avoid matching, e.g. "a12" if "a1" is searched
                return (s1 + ".") in (s2 + ".")
        else:
            return s1 in s2
    else:
        return False


def matchlst(string, strlst, exact=False):
    return [s for s in strlst if matchstr(string, s.lower(), exact)]


def matchdict(value, key, dct, negate=False, exact=False):
    if key in dct and value:
        return value != dct[key] if negate else value == dct[key]
    else:
        return not exact


def matchop(op, value, query, exact=False):
    if query:
        if "not" != op:
            if op:
                result = eval(op)(
                    matchstr(q, value.lower(), exact) for q in query
                )
            else:  # any
                result = any(matchstr(q, value.lower(), exact) for q in query)
        else:  # not
            result = all(not matchstr(q, value.lower(), exact) for q in query)
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
        string,
    )


def parselog(
    database, strbuild, jobname, infokey, info, txt, nentries, nerrors
):
    pattern = (
        r"^\+\+\+ PERFORMANCE ([a-zA-Z]+(?:[0-9_\-,]+[a-zA-Z]+)*)([^\+\-]+)"
    )
    matches = [
        match
        for match in re.finditer(pattern, txt, re.MULTILINE | re.DOTALL)
        if match and match.group(1) and match.group(2)
    ]
    m = n = 0
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
                    m = m + 1
                database[strbuild][category][jobname] = values
            else:
                n = n + 1
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
                            m = m + 1
                        database[strbuild][category][i] = values[i]
                else:
                    if jobname not in database[strbuild][category]:
                        m = m + 1
                    database[strbuild][category][jobname] = values
            else:
                n = n + 1
    if (infokey and info) and (
        strbuild in database and infokey not in database[strbuild]
    ):
        database[strbuild][infokey] = info
        if 0 == m:  # ensure database rewrite
            m = m + 1
    return nentries + m, nerrors + n


def fname(extlst, in_main, in_dflt, idetail=""):
    """
    Build filename from components and list of file-extensions.
    """
    dflt, inplst = pathlib.Path(in_dflt), str(in_main).strip().split()
    path, result = pathlib.Path(inplst[0] if inplst else in_main), []
    if path.is_dir():
        figext = [dflt.suffix[1:]] if 1 >= len(inplst) else inplst[1:]
        result = [
            path / f"{dflt.stem}{idetail}.{ext}"
            for ext in figext
            if ext in extlst
        ]
    elif path.suffix[1:] in extlst:
        result = [path.parent / f"{path.stem}{idetail}{path.suffix}"]
    elif "." == str(path.parent):
        figext = f".{path.name}" if path.name in extlst else dflt.suffix
        result = [path.parent / f"{dflt.stem}{idetail}{figext}"]
    else:
        figstm = path.stem if path.stem else dflt.stem
        result = [path.parent / f"{figstm}{idetail}{dflt.suffix}"]
    return result


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
            if ".json" == filename.suffix.lower():
                with os.fdopen(tmpfile[0], "w") as file:
                    json.dump(sortdb(database), file, indent=2)
                    file.write("\n")  # append newline at EOF
            else:  # pickle
                with os.fdopen(tmpfile[0], "wb") as file:
                    pickle.dump(sortdb(database), file)
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
    """
    Rounds a number to the given number of decimals.
    """
    dec = pow(10, decimals)
    nom = int((dec * num + 0.5) if 0 <= num else (dec * num - 0.5))
    return nom / dec if 1 != dec else nom


def num2str(num):
    return str(num).rstrip("0").rstrip(".")


def divup(a, b):
    return int((a + b - 1) / b)


def trend(values):
    """
    Calculate the predicted value (linear trend of history),
    the relative difference of lastest and previous value (rd),
    the standard deviation (cv), and the linear trend (equation).
    """
    rd, cv, eqn, size = None, None, None, len(values)
    b = values[0] if 0 < size else 0
    a = values[1] if 1 < size else 0
    if 0 != a:
        rd = (b - a) / a
    if 2 < size:
        # b: predicted value for x=0 (a * x + b)
        a, b = numpy.polyfit(range(1, size), values[1:], deg=1)
        eqn = numpy.poly1d((a, b))
        avg = numpy.mean(values[1:])
        if 0 != avg:
            cv = numpy.std(values[1:]) / avg
    return (b, rd, cv, eqn)


def bold(s, cond=True):
    if cond:
        c, t = s.count("$"), s.replace("%", r"\%")
        return r"$\bf{" + (t.replace("$", "") if 0 == (c % 2) else t) + "}$"
    else:
        return s


def conclude(values, base, unit, accuracy, bounds, lowhigh):
    label, bad = f"{num2fix(values[0], accuracy)} {unit}", False
    guess, rd, cv, eqn = trend(values)  # unpack
    blist = base.split()
    if 1 < len(blist):  # category and detail
        dlist = blist[1].split("_")
        for c in blist[0].split("_"):
            while c in dlist:  # no redundancy
                dlist.remove(c)
        base = f"{blist[0]} {'_'.join(dlist)}"
    if rd:
        inum = num2fix(100 * rd)
        if cv and bounds and 0 != bounds[0]:
            anum = f"{inum}%" if 0 <= inum else f"|{inum}%|"
            bnum = num2fix(max(100 * cv, 1))
            cnum = num2fix(abs(bounds[0]), accuracy)
            t0 = num2fix(bnum * abs(bounds[0]))
            t1 = num2fix(abs(bounds[1])) if 1 < len(bounds) else 0
            cond = f"min({bnum}%*{num2str(cnum)}, {t1}%)"
            if t0 < t1 or 0 >= t1:
                if t0 < abs(inum):
                    bad = (0 > inum and lowhigh[0]) or (
                        0 < inum and lowhigh[1]
                    )
                    btext = bold(label, bad or not (lowhigh[0] or lowhigh[1]))
                    label = f"{base} = {btext}  {anum}>{cond}"
                else:
                    expr = f"{anum}" + r"$\leq$" + cond
                    label = f"{base} = {label}  {expr}"
            else:
                if t1 < abs(inum):
                    bad = (0 > inum and lowhigh[0]) or (
                        0 < inum and lowhigh[1]
                    )
                    btext = bold(label, bad or not (lowhigh[0] or lowhigh[1]))
                    label = f"{base} = {btext}  {anum}>{cond}"
                else:
                    expr = f"{anum}" + r"$\leq$" + cond
                    label = f"{base} = {label}  {expr}"
        else:
            sign = ("+" if 0 < inum else "") if 0 != inum else r"$\pm$"
            label = f"{base} = {label}  {sign}{inum}%"
    else:
        label = f"{base} = {label}"
    return label, bad, eqn


def create_figure(plots, nplots, resint, untied, addon):
    figsize = (divup(resint[0], resint[2]), divup(resint[1], resint[2]))
    subplots = plot.subplots(
        max(nplots, 1),
        sharex=True,
        figsize=figsize,
        dpi=resint[2],
    )
    figure, axes = subplots  # unpack
    if 2 > nplots:  # ensure axes object is always a list
        axes = [axes]
    i = 0
    for entry in plots:
        for data in plots[entry]:
            axes[i].step(data[0], ".:", where="mid", label=data[1])
            axes[i].set_ylabel(f"{entry.upper()} [{data[2]}]")
            axes[i].xaxis.set_ticks(
                range(len(data[-1])), data[-1], rotation=45
            )
            if 1 < len(data[-1]):
                axes[i].set_xlim(0, len(data[-1]) - 1)  # tighter bounds
            axes[i].legend(loc="upper left", fontsize="small")  # ncol=2
            if untied:
                i = i + 1
        if not untied:
            i = i + 1
    if 0 < nplots:
        axes[-1].set_xlabel("Build Number")
        title = "Performance History"
        figure.suptitle(
            f"{title} ({addon.lower()})" if addon else title,
            fontsize="x-large",
        )
        figure.tight_layout(rect=[0, 0, 1, 0.98])  # fit suptitle
        figure.gca().invert_xaxis()
    return figure


def main(args, argd, dbfname):
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
    qlst = rslt.split(",")
    nerrors, nentries, accuracy = 0, 0, 1
    inflight = max(args.inflight, 0)
    info = {"branch": args.branch} if args.branch else {}
    infokey = "INFO"
    outfile = None
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
            if dbfname == argd.filepath and args.infile.is_file()
            else dbfname
        )
    elif args.infile is None:  # connect to URL
        outfile = dbfname

    # timestamp before loading database
    ofmtime = mtime(outfile)
    database = loaddb(dbfname)
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
    write = False
    for entries in (
        build[b] for build in database.values() for b in build if infokey != b
    ):
        for key, entry in entries.items():
            name = parsename(key)
            if isinstance(entry, dict):  # JSON-format
                for e in entry:
                    if name not in weights:
                        weights[name] = {}
                        weights[name][e] = 1.0
                        write = True
                    elif e not in weights[name]:
                        weights[name][e] = 1.0
                        write = True
            elif name not in weights:  # telegram format
                write = [1.0 for e in entry if ":" in e]
                if write:
                    weights[name] = write
    if write:  # write weights if modified
        savedb(args.weights, weights, wfmtime, 3)

    nbuilds, nbuild = 0, int(args.nbuild) if args.nbuild else 0
    if args.infile and (args.infile.is_file() or args.infile.is_fifo()):
        next = latest + 1
        nbld = nbuild if 0 < nbuild else next
        name = (
            args.query
            if args.query and (args.query != argd.query or args.query_exact)
            else args.infile.stem
        )
        nentries, nerrors = parselog(
            database, str(nbld), name, infokey, info, txt, nentries, nerrors
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

        njobs = 0
        while builds:
            # iterate over all builds (latest first)
            for build in builds:
                ibuild = build["number"] if "number" in build else 0
                strbuild = str(ibuild)  # JSON stores integers as string
                if (  # consider early exit
                    ibuild <= max(latest - inflight, 1)
                    and matchdict("running", "state", build, negate=True)
                    and strbuild in database
                ):
                    latest, builds = ibuild, None
                    break
                m, n = nentries, 0
                infocpy = info
                if "branch" in build and matchdict(
                    build["branch"], "branch", info, negate=True
                ):
                    infocpy = copy.deepcopy(info)
                    infocpy["branch"] = build["branch"]
                jobs = build["jobs"] if "jobs" in build else []
                for job in (
                    job
                    for job in jobs
                    if "name" in job and matchdict(0, "exit_status", job)
                ):
                    if 2 <= abs(args.verbosity):
                        if 0 == n:
                            print(f"[{ibuild}]", end="", flush=True)
                        print(".", end="", flush=True)
                    log = (
                        requests.get(job["log_url"], headers=auth).text
                        if "log_url" in job
                        else ""
                    )
                    txt = json.loads(log) if log else {}
                    if txt and "content" in txt:
                        nentries, nerrors = parselog(
                            database,
                            strbuild,
                            job["name"],
                            infokey,
                            infocpy,
                            txt["content"],
                            nentries,
                            nerrors,
                        )
                        n = n + 1
                if 0 < n:
                    nbuilds = nbuilds + (1 if m != nentries else 0)
                    njobs = njobs + n
                if (  # consider early exit
                    (0 == n and ibuild <= latest)
                    and matchdict("running", "state", build, negate=True)
                ) or (args.history <= nbuilds or nbuild == ibuild):
                    latest, builds = ibuild, None
                    break
            if builds and 1 < ibuild and "page" in params:
                params["page"] = params["page"] + 1  # next page
                builds = requests.get(url, params=params, headers=auth).json()
            else:
                builds = None
        if 2 <= abs(args.verbosity) and 0 < njobs:
            print("[OK]")

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
        savedb(outfile, database, ofmtime, 3)
        if 2 <= abs(args.verbosity) and outfile and not outfile.exists():
            print(f"{outfile} database created.")

    # conclude loading data from latest CI
    if 2 <= abs(args.verbosity):
        if 0 != nerrors:
            y = "ies" if 1 != nerrors else "y"
            print(
                f"WARNING: ignored {nerrors} erroneous entr{y}!",
                file=sys.stderr,
            )
        y = "ies" if 1 != nentries else "y"
        print(f"Database consists of {dbsize} builds.")
        if 0 < nentries:
            s = "s" if 1 != nbuilds else ""
            print(f"Found {nentries} new entr{y} in {nbuilds} build{s}.")
        else:
            print(f"Found {nentries} new entr{y}.")

    if dbkeys and args.figure:  # determine template-record for figure
        if nbuild in dbkeys:
            templkey = dbkeys[nbuild]
        elif not args.infile or not (
            args.infile.is_file() or args.infile.is_fifo()
        ):  # find template with most content
            templkeys = dbkeys[-min(inflight + 1, dbsize) :]  # noqa: E203
            templkey, s = None, 0
            for k in templkeys:
                t = sum(len(v) for v in database[k].values())
                if s <= t:
                    templkey, s = k, t
        else:  # file-based input (just added)
            templkey = dbkeys[-1]
        template = database[templkey]
    else:
        template = dict()

    entries = [  # collect all categories
        e  # category (one level below build number)
        for e in template
        if (e != infokey or (select and infokey in select))
        and matchop("any", e, select, args.select_exact)
    ]
    if entries and not select and args.select_exact:
        entries = [entries[-1]]  # assume insertion order is preserved

    # parse bounds used to highlight results
    strbounds = re.split(
        r"[\s;,]", (args.bounds if args.bounds else argd.bounds).strip()
    )
    try:
        highlight = [float(i) for i in strbounds]
    except ValueError:
        highlight = [float(i) for i in re.split(r"[\s;,]", argd.bounds)]
    lowhigh = (  # meaning of negative/positive deviation
        highlight
        and 0 != highlight[0]
        and "" != strbounds[0]
        and "-" == strbounds[0][0],
        highlight
        and 0 != highlight[0]
        and "" != strbounds[0]
        and "+" == strbounds[0][0],
    )
    exceeded = False

    # build figure
    query_op = args.query_op if args.query_op else argd.query_op
    transpat = "!\"#$%&'()*+-./:<=>?@[\\]^_`{|}~"
    split = str.maketrans(transpat, " " * len(transpat))
    clean = str.maketrans("", "", transpat)
    plots, sval, yunit, addon = {}, None, None, args.branch
    ngraphs = span = 0
    for entry in entries:
        n = 0
        for value in (
            v for v in template[entry] if matchop(query_op, v, query)
        ):
            wname = value.split()[0]  # name/key of weight-entry
            wlist = weights[wname] if wname in weights else []
            layers, xvalue, yvalue = dict(), [], []
            legend, aunit = value, None
            if value not in match:
                match.append(value)
            builds = [
                b  # collect builds to be plotted
                for b in database
                if (entry in database[b] and value in database[b][entry])
                and (  # match branch
                    not (infokey and args.branch)
                    or infokey not in database[b]
                    or "branch" not in database[b][infokey]
                    or (database[b][infokey]["branch"] == args.branch)
                )
            ]
            # collect common keys (if inline-JSON)
            keys = []
            for build in builds:
                values = database[build][entry][value]
                if isinstance(values, dict):  # JSON-format
                    for key in matchlst(qlst[0], values.keys(), args.exact):
                        if key not in keys:
                            keys.append(key)
                else:
                    break
            # collect data to be plotted
            for build in reversed(builds):  # order: latest -> older
                values, ylabel = database[build][entry][value], None
                if isinstance(values, dict):  # JSON-format
                    vals, legd = [], []
                    for key in (k for k in keys if k in values):
                        vscale = 1.0 if 2 > len(qlst) else float(qlst[1])
                        weight = wlist[key] if key in wlist else 1.0
                        strval = str(values[key])  # ensure string
                        parsed = parseval(strval)
                        unit = (
                            strval[parsed.end(3) :].strip()  # noqa: E203
                            if parsed
                            else ""
                        )
                        vals.append(float(strval.split()[0]) * vscale * weight)
                        if not ylabel:
                            ylabel = (
                                (unit if unit else key)
                                if 3 > len(qlst)
                                else qlst[2]
                            )
                        lst = key.translate(split).split()
                        detail = [s for s in lst if s.lower() != qlst[0]]
                        legd.append(
                            f"{value} {'_'.join(detail)}" if detail else value
                        )
                    if vals:
                        if yvalue:
                            if not isinstance(yvalue[0], list) or (
                                len(yvalue[0]) == len(vals)
                            ):  # same dimensionality
                                yvalue.append(vals)
                        elif int(build) == latest:
                            yvalue, legend = [vals], legd
                            if addon == args.branch:
                                addon = rslt.split(",")[0] + (
                                    f"@{addon}" if addon else ""
                                )  # title-addon
                        xvalue.append(int(build))
                else:  # telegram format
                    # match --result primarily against "unit"
                    for v in reversed(values):  # match last entry
                        parsed = parseval(v)
                        if parsed and parsed.group(3):
                            unit = v[parsed.end(3) :].strip()  # noqa: E203
                            ulow = unit.lower()
                            if not ylabel and matchstr(rslt, ulow):
                                yvalue.append(float(parsed.group(3)))
                                xvalue.append(int(build))
                                ylabel = unit
                    # match --result secondary against "init"
                    for v in reversed(values):  # match last entry
                        parsed = parseval(v)
                        if parsed and parsed.group(3):
                            init = (
                                parsed.group(1).strip(": ")
                                if parsed.group(1)
                                else ""
                            )
                            unit = v[parsed.end(3) :].strip()  # noqa: E203
                            ulab = unit if unit else init
                            ilow = init.lower()
                            if not ylabel and matchstr(rslt, ilow):
                                yvalue.append(float(parsed.group(3)))
                                xvalue.append(int(build))
                                ylabel = ulab
                            if init and (not aunit or ulab == aunit):
                                if init not in layers:
                                    if not aunit:
                                        aunit = ulab
                                    layers[init] = []
                                layers[init].append(float(parsed.group(3)))

            wdflt, s, j = True, args.history, 0
            # trim, and apply weights
            for a in layers.keys():
                y = layers[a]
                s = min(s, len(y))
                w = wlist[j] if j < len(wlist) else 1.0
                if 1.0 != w:
                    layers[a] = [y[k] * w for k in range(s)]
                    wdflt = False  # non-default weight discovered
                else:  # unit-weight
                    layers[a] = [y[k] for k in range(s)]
                j = j + 1
            if not yunit and (ylabel or args.result):
                yunit = (ylabel if ylabel else args.result).split()[0]
            # summarize layer into yvalue only in case of non-default weights
            if (not aunit or aunit == yunit) and not wdflt:
                yvalue = [sum(y) for y in zip(*layers.values())]
            elif yvalue:  # trim
                yvalue = yvalue[0 : args.history]  # noqa: E203
            xsize = len(yvalue) if yvalue else 0
            xvalue = xvalue[0:xsize]  # trim

            if 0 < xsize:  # skip empty plot
                # perform some trend analysis
                eqn = None
                if isinstance(legend, list):
                    ylist, ylabel = list(zip(*yvalue)), []
                    for j in range(len(legend)):
                        y, z = ylist[j], legend[j]
                        label, bad, eqn = conclude(
                            y, z, yunit, accuracy, highlight, lowhigh
                        )
                        ylabel.append(label)
                        if not exceeded and bad:
                            exceeded = True
                    ylabel = ylabel if 1 < len(ylabel) else ylabel[0]
                else:
                    ylabel, bad, eqn = conclude(
                        yvalue, legend, yunit, accuracy, highlight, lowhigh
                    )
                    if not exceeded and bad:
                        exceeded = True
                # plot values and legend as collected above
                if (not aunit or aunit == yunit) and (
                    yvalue and latest == xvalue[0]
                ):
                    ispan = xsize * xsize / (xvalue[0] - xvalue[-1] + 1)
                    if span < ispan:
                        sval, span = xvalue, ispan
                    # collect plot data
                    if entry not in plots:
                        plots[entry] = []
                    plots[entry].append([yvalue, ylabel, yunit, sval])
                    n = n + 1
        # maximum number of graphs discovered over all plot-areas
        ngraphs = max(ngraphs, n)

    # determine image resolution
    rdef = [int(r) for r in argd.resolution.split("x")]
    if 2 == len(rdef):
        rdef.append(100)
    rstr, resint = args.resolution.split("x"), []
    for i in range(len(rdef)):
        try:
            resint.append(int(rstr[i]))
        except:  # noqa: E722
            r = rdef[i] if 1 != i else round(resint[0] * rdef[1] / rdef[0])
            resint.append(r)

    nplots = len(plots)
    if 0 < nplots:
        nplots_untied = sum(len(v) for v in plots.values())
        # auto-adjust y-resolution according to number of plots
        if args.resolution == argd.resolution:  # resolution not user-defined
            resint_untied = copy.deepcopy(resint)
            resint_untied[1] = resint[2] * divup(
                resint[1] * math.sqrt(nplots_untied), resint[2]
            )
        else:  # alias
            resint_untied = resint

        # setup primary figure
        if args.untied:
            nplots_primry, resint_primry = nplots_untied, resint_untied
        else:
            nplots_primry, resint_primry = nplots, resint
        figure_primry = create_figure(
            plots, nplots_primry, resint_primry, args.untied, addon
        )

        # supported file types and filename components
        figdet = (
            ""  # eventually add details about category
            if 1 < len(entries) or 0 == len(entries)
            else f"-{entries[0].translate(clean)}"
        )
        figcat = re.sub(r"[ ,;]+", "_", figdet)
        if 0 < len(match) and 2 >= len(match):
            match = [re.sub(r"[ ,;]+", "_", s.translate(clean)) for s in match]
            parts = [s.lower() for c in match for s in c.split("_")]
            figqry = f"-{'_'.join(dict.fromkeys(parts))}{figcat}"
        elif query:
            figqry = f"-{'_'.join(query)}{figcat}"
        else:
            figqry = figcat
        figout = fname(
            extlst=figure_primry.canvas.get_supported_filetypes().keys(),
            in_main=args.figure,
            in_dflt=argd.figure,
            idetail=figqry,
        )

        # setup untied figure
        if 1 < len(figout) and not args.untied:
            figure_untied = create_figure(
                plots, nplots_untied, resint_untied, True, addon
            )
        else:  # alias
            resint_untied = resint_primry
            figure_untied = figure_primry

        # save figure(s) for all requested formats
        for i in range(len(figout)):
            figure = figure_primry if 0 == i else figure_untied
            figres = resint_primry if 0 == i else resint_untied
            # reduce file size (png) and save figure
            if ".png" == figout[i].suffix.lower():
                figure.canvas.draw()  # otherwise the image is empty
                imageraw = figure.canvas.tostring_rgb()
                image = PIL.Image.frombytes("RGB", figres[0:2], imageraw)
                # avoid Palette.ADAPTIVE, consider back/foreground color
                image = image.convert("P", colors=ngraphs + 2)
                image.save(figout[i], "PNG", optimize=True)
            else:
                figure.savefig(figout[i])  # save graphics file
            if 1 == abs(args.verbosity) and 0 == i:  # notify only first
                print(f"{figout[i]} created.")

    return exceeded


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
        default="1600x900",
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
        "--branch",
        type=str,
        default=None,
        help="Branch, all if None",
    )
    argparser.add_argument(
        "-c",
        "--organization",
        type=str,
        default="intel",
        help="Buildkite org/slug/company",
    )
    argparser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        default="tpp-libxsmm",
        help="Buildkite pipeline",
    )
    argparser.add_argument(
        "-q",
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
        help="Category, all if None",
    )
    argparser.add_argument(
        "-r",
        "--result",
        type=str,
        default="ms",
        help="Plotted values",
    )
    argparser.add_argument(
        "-e",
        "--exact",
        action="store_true",
        help="Match result exactly",
    )
    argparser.add_argument(
        "-t",
        "--bounds",
        type=str,
        default="2.0 10",
        help="Highlight if exceeding max(A*Stdev%%,B%%)",
    )
    argparser.add_argument(
        "-u",
        "--untied",
        action="store_true",
        help="Separate plot per query",
    )
    argparser.add_argument(
        "-m",
        "--inflight",
        type=int,
        default=2,
        help="Rescan builds",
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

    args = argparser.parse_args()  # 1st pass
    if args.pipeline:
        filepath = rdir / f"{args.pipeline}.json"
        figure = f"{args.pipeline}.{figtype}"
        argparser.set_defaults(filepath=filepath, figure=figure)
        args = argparser.parse_args()  # 2nd pass
    argd = argparser.parse_args([])
    dbfname = fname(  # database filename
        ["json", "pickle", "pkl", "db"],
        in_main=args.filepath,
        in_dflt=argd.filepath,
    )
    if dbfname and dbfname[0].name:
        weights = dbfname[0].with_name(
            f"{dbfname[0].stem}.weights{dbfname[0].suffix}"
        )
        argparser.set_defaults(weights=weights)
        args = argparser.parse_args()  # 3rd pass
    argd = argparser.parse_args([])

    exceeded = main(args, argd, dbfname[0] if dbfname else None)
    if exceeded:
        if 2 <= abs(args.verbosity):
            print(
                "WARNING: deviation of latest value exceeds margin.",
                file=sys.stderr,
            )
        exit(1)
