#!/usr/bin/env python3
import requests
import json
import sys
import re

argc = len(sys.argv)
if 3 < argc:
    url = sys.argv[1]
    auth_token = sys.argv[2]
    filename = sys.argv[3]
else:
    sys.tracebacklimit = 0
    raise ValueError(sys.argv[0] + ": please pass URL, TOKEN, and FILENAME!")

auth_param = {"Authorization": "Bearer {}".format(auth_token)}
params = {"per_page": 100, "page": 1}

try:  # proceeed with cached results in case of an error
    builds = requests.get(url, params=params, headers=auth_param).json()
except:  # noqa: E722
    print("WARNING: failed to connect to {}\n".format(url))
    builds = None
    pass

try:
    with open(filename, "r") as file:
        database = json.load(file)
except:  # noqa: E722
    database = dict()
    pass

nerrors = 0
nentries = 0
for build in builds:
    nbuild = build["number"]
    # JSON stores integers as string
    sbuild = str(nbuild)
    if sbuild in database:
        break
    print("|", end="", flush=True)
    for job in (job for job in build["jobs"] if 0 == job["exit_status"]):
        print(".", end="", flush=True)
        log = requests.get(job["log_url"], headers=auth_param)
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

with open(filename, "w") as file:
    json.dump(database, file, indent=2)
    file.write("\n")  # append newline at EOF
