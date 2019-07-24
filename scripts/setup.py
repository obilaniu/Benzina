# -*- coding: utf-8 -*-

#
# Imports and early Python version check.
#
packageName = "benzina"
githubURL   = "https://github.com/obilaniu/Benzina"

import os, sys
if sys.version_info[:2] < (3, 5):
    sys.stdout.write(packageName+" is Python 3.5+ only!\n")
    sys.exit(1)
from setuptools import setup, find_packages, Extension
from .          import git, versioning, utils


#
# Read long description
#
with open(os.path.join(git.getSrcRoot(),
                       "scripts",
                       "LONG_DESCRIPTION.txt"), "r", encoding="utf-8") as f:
    long_description = f.read()


#
# Synthesize version.py file
#
with open(os.path.join(git.getSrcRoot(),
                       "src",
                       packageName,
                       "version.py"), "w") as f:
    f.write(versioning.synthesizeVersionPy())

author = "Olexa Bilaniuk"


#
# Perform setup.
#
setup(
    name                 = packageName,
    version              = versioning.verPublic,
    author               = author,
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
        "meson>=0.51.1",
    ],
    install_requires     = [
        "meson>=0.51.1",
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
                                             "lib")],
                  runtime_library_dirs=[os.path.join("$ORIGIN", "lib")],
                  libraries=["benzina"],)
    ],
    cmdclass={
        "build_configure": utils.build_configure,
        "build_ext":       utils.build_ext,
        "clean":           utils.clean,
    },
    command_options={
        'build_sphinx': {
            'project': ("setup.py", packageName),
            'copyright': ("setup.py", "2019, {}".format(author)),
            'version': ("setup.py", versioning.verRelease),
            'release': ("setup.py", versioning.verPublic)
        }
    },
    zip_safe             = False,
)
