# -*- coding: utf-8 -*-

#
# Imports
#
import os, sys, subprocess, time
from   setuptools import setup, find_packages, Extension

packageName = "benzina"
githubURL   = "https://github.com/obilaniu/Benzina"


#
# Restrict to Python 3.5+
#
if sys.version_info[:2] < (3, 5):
	sys.stdout.write(packageName+" is Python 3.5+ only!\n")
	sys.exit(1)


#
# Retrieve setup scripts
#
from . import git, versioning, utils


#
# Read long description
#
with open(os.path.join(git.getSrcRoot(),
                       "scripts",
                       "LONG_DESCRIPTION.txt"), "r") as f:
	long_description = f.read()


#
# Synthesize version.py file
#
with open(os.path.join(git.getSrcRoot(),
                       "src",
                       packageName,
                       "version.py"), "w") as f:
	f.write(versioning.synthesizeVersionPy())



#
# Perform setup.
#
setup(
    name                 = packageName,
    version              = versioning.verPublic,
    author               = "Olexa Bilaniuk",
    author_email         = "anonymous@anonymous.com",
    license              = "MIT",
    url                  = githubURL,
    download_url         = githubURL+"/archive/v{}.tar.gz".format(versioning.verRelease),
    description          = "A fast image-loading package to load images compressed with "
                           "video codecs onto GPU asynchronously.",
    long_description     = long_description,
    classifiers          = [
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    python_requires      = '>=3.5',
    setup_requires       = [
        "meson>=0.45",
    ],
    install_requires     = [
        "nauka>=0.0.8",
        "numpy>=1.10",
    ],
    packages             = find_packages("src"),
    package_dir          = {'': 'src'},
    ext_modules          = [
        Extension("benzina.native",
                  [os.path.join("src", "benzina", "native.c")],
                  include_dirs=[os.path.join(git.getSrcRoot(), "include")],
                  library_dirs=[os.path.join(git.getSrcRoot(),
                                             utils.get_build_platlib(),
                                             "benzina",
                                             "libs")],
                  runtime_library_dirs=[os.path.join("$ORIGIN", "libs")],
                  libraries=["benzina"],)
    ],
    cmdclass={
        "build_ext": utils.build_ext,
        "clean":     utils.clean,
    },
    zip_safe             = False,
)
