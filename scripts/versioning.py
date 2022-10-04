import os, re, time
from pathlib import Path
from .       import git


#
# Public Version.
#
# VERSION.txt in combination with this file contain the master declaration of
# the version number for this project.
#
# We will obey PEP 440 (https://www.python.org/dev/peps/pep-0440/) here. PEP440
# recommends the pattern
#     [N!]N(.N)*[{a|b|rc}N][.postN][.devN]
# We shall standardize on the ultracompact form
#     [N!]N(.N)*[{a|b|rc}N][-N][.devN]
# which has a well-defined normalization.
#

ver_release = open(Path(__file__).with_name('VERSION.txt')).read().strip()
ver_public  = ver_release

#
# Information computed from the public version.
#
regex_match = re.match(r"""(?:
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
)""", ver_public, re.X)
assert regex_match
ver_epoch   = regex_match.group("epoch")   or ""
ver_prerel  = regex_match.group("pre")     or ""
ver_postrel = regex_match.group("post")    or ""
ver_devrel  = regex_match.group("dev")     or ""
ver_normal  = ver_release+ver_prerel+ver_postrel+ver_devrel
ver_is_rel  = bool(not ver_prerel and not ver_devrel)

#
# Local Version.
#
# Uses POSIX time (Nominal build time as seconds since the Epoch) as obtained
# either from the environment variable SOURCE_DATE_EPOCH or the wallclock time.
# Also converts POSIX timestamp to ISO 8601.
#
ver_vcs       = git.get_git_ver()
ver_clean     = bool((not ver_vcs) or (git.is_git_clean()))
posix_time    = int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
iso_8601_time = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(posix_time))
ver_local     = ver_public+"+"+iso_8601_time
if ver_vcs:
    ver_local += "."+ver_vcs
    if not ver_clean:
        ver_local += ".dirty"

#
# SemVer Version.
#
# Obeys Semantic Versioning 2.0.0, found at
#     https://semver.org/spec/v2.0.0.html
#
ver_semver  = ".".join((ver_release+".0.0").split(".")[:3])
identifiers= []
if ver_prerel: identifiers.append(ver_prerel)
if ver_devrel: identifiers.append(ver_devrel[1:])
if identifiers:
    ver_semver += "-" + ".".join(identifiers)
metadata   = []
if regex_match.group("postN"):
    metadata.append("post")
    metadata.append(regex_match.group("postN"))
metadata.append("buildtime")
metadata.append(iso_8601_time)
if ver_vcs:
    metadata.append("git")
    metadata.append(ver_vcs)
    if not ver_clean:
        metadata.append("dirty")
if metadata:
    ver_semver += "+" + ".".join(metadata)


#
# Version utilities
#
def synthesize_version_py():
    with open(Path(__file__).with_name('version.py.in')) as f:
        return f.read().format(**globals())
