#
# Imports and early Python version check.
#
package_name = "benzina"
github_url   = "https://github.com/obilaniu/Benzina"
author       = "Olexa Bilaniuk"

import glob, os, sys
if sys.version_info[:2] < (3, 6):
    sys.stdout.write(package_name+" is Python 3.6+ only!\n")
    sys.exit(1)
from setuptools import setup, find_packages, Extension
from .          import git, versioning, utils


#
# Read long description
#
with open(os.path.join(git.get_src_root(),
                       "scripts",
                       "LONG_DESCRIPTION.txt"), "r", encoding="utf-8") as f:
    long_description = f.read()


#
# Synthesize version.py file
#
with open(os.path.join(git.get_src_root(),
                       "src",
                       package_name,
                       "version.py"), "w") as f:
    f.write(versioning.synthesize_version_py())



#
# Perform setup.
#
setup(
    name                 = package_name,
    version              = versioning.ver_public,
    author               = author,
    author_email         = "anonymous@anonymous.com",
    license              = "MIT",
    url                  = github_url,
    download_url         = github_url+"/archive/v{}.tar.gz".format(versioning.ver_release),
    description          = "A fast image-loading package to load images compressed with "
                           "video codecs onto GPU asynchronously.",
    long_description     = long_description,
    classifiers          = [
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    zip_safe             = False,
    python_requires      = '>=3.6',
    install_requires     = [
        "numpy>=1.10",
        "pytest>=6.0.1",
        "bcachefs>=0.1.16",
        "pybenzinaparse>=0.2.2",
    ],
    extras_require       = {
        "coco": [
            "torchvision",
            "pycocotools>=2.0.5"
        ],
    },
    packages             = find_packages("src"),
    package_dir          = {'': 'src'},
    ext_modules          = [
        Extension("benzina.native",
                  [os.path.join("src", "benzina", "native.c")],
                  include_dirs=[os.path.join(git.get_src_root(), "include")],
                  runtime_library_dirs=[os.path.join("$ORIGIN", "lib")],
                  define_macros=[("PY_SSIZE_T_CLEAN", None)],
                  extra_compile_args=['-Wno-cpp'],
                  libraries=["benzina"],
        ),
    ],
    cmdclass={
        "develop":         utils.develop,
        "install":         utils.install,
        "build_configure": utils.build_configure,
        "build_ext":       utils.build_ext,
        "clean":           utils.clean,
        "meson_test":      utils.meson_test,
    },
    command_options={
        'build_configure': {'meson_require': ("setup.py", "meson>=0.64.0")},
        'build_sphinx':    {
            'project':   ("setup.py", package_name),
            'copyright': ("setup.py", "2022, {}".format(author)),
            'version':   ("setup.py", versioning.ver_release),
            'release':   ("setup.py", versioning.ver_public)
        },
    },
)
