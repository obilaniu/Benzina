#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, sys, subprocess, time
from   . import git

#
# Public Version.
#
# This is the master declaration of the version number for this project.
#
# We will obey PEP 440 (https://www.python.org/dev/peps/pep-0440/) here. PEP440
# recommends the pattern
#     [N!]N(.N)*[{a|b|rc}N][.postN][.devN]
# We shall standardize on the ultracompact form
#     [N!]N(.N)*[{a|b|rc}N][-N][.devN]
# which has a well-defined normalization.
#

verPublic  = "0.0.1"

#
# Information computed from the public version.
#
regexMatch = re.match(r"""(?:
    (?:(?P<epoch>[0-9]+)!)?               # epoch
    (?P<release>[0-9]+(?:\.[0-9]+)*)      # release segment
    (?P<pre>                              # pre-release
        (?P<preL>a|b|rc)
        (?P<preN>[0-9]+)
    )?
    (?P<post>                             # post release
        (?:-(?P<postN>[0-9]+))
    )?
    (?P<dev>                              # dev release
        (?:\.dev(?P<devN>[0-9]+))
    )?
)""", verPublic, re.X)
assert regexMatch
verEpoch   = regexMatch.group("epoch")   or ""
verRelease = regexMatch.group("release")
verPreRel  = regexMatch.group("pre")     or ""
verPostRel = regexMatch.group("post")    or ""
verDevRel  = regexMatch.group("dev")     or ""
verNormal  = verRelease+verPreRel+verPostRel+verDevRel
verIsRel   = bool(not verPreRel and not verDevRel)

#
# Local Version.
#
# Uses POSIX time (Nominal build time as seconds since the Epoch) as obtained
# either from the environment variable SOURCE_DATE_EPOCH or the wallclock time.
# Also converts POSIX timestamp to ISO 8601.
#
verVCS     = git.getGitVer()
verClean   = bool((not verVCS) or (git.isGitClean()))
posixTime  = int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
iso8601Time= time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(posixTime))
verLocal   = verPublic+"+"+iso8601Time
if verVCS:
	verLocal += "."+verVCS
	if not verClean:
		verLocal += ".dirty"

#
# SemVer Version.
#
# Obeys Semantic Versioning 2.0.0, found at
#     https://semver.org/spec/v2.0.0.html
#
verSemVer  = ".".join((verRelease+".0.0").split(".")[:3])
identifiers= []
if verPreRel: identifiers.append(verPreRel)
if verDevRel: identifiers.append(verDevRel[1:])
if identifiers:
	verSemVer += "-" + ".".join(identifiers)
metadata   = []
if regexMatch.group("postN"):
	metadata.append("post")
	metadata.append(regexMatch.group("postN"))
metadata.append("buildtime")
metadata.append(iso8601Time)
if verVCS:
	metadata.append("git")
	metadata.append(verVCS)
	if not verClean:
		metadata.append("dirty")
if metadata:
	verSemVer += "+" + ".".join(metadata)


#
# Version utilities
#
def synthesizeVersionPy():
	templatePath = os.path.join(git.getSrcRoot(),
	                            "scripts",
	                            "version.py.in")
	
	with open(templatePath, "r") as f:
		return f.read().format(**globals())
