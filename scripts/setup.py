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
        "Development Status :: 1 - Planning",
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
    setup_requires       = [
        "meson>=0.54.0",
    ],
    install_requires     = [
        "meson>=0.54.0",
        "numpy>=1.10",
        "pybenzinaparse @ git+https://github.com/satyaog/pybenzinaparse.git@be4c7784f488100269ae991a340fcb15b65c20f9#egg=pybenzinaparse-0.2.1",
    ],
    packages             = find_packages("src"),
    package_dir          = {'': 'src'},
    ext_modules          = [
        Extension("benzina.native",
                  [os.path.join("src", "benzina", "native.c")],
                  include_dirs=[os.path.join(git.get_src_root(), "include")],
                  library_dirs=[os.path.join(git.get_src_root(),
                                             utils.get_build_platlib(),
                                             "benzina",
                                             "lib")],
                  runtime_library_dirs=[os.path.join("$ORIGIN", "lib")],
                  libraries=["benzina"],
        ),
        Extension("benzina._native",
                  glob.glob(os.path.join("src", "benzina", "_native", "**", "*.c"),
                            recursive=True),
                  include_dirs=[os.path.join(git.get_src_root(), "include")],
                  library_dirs=[os.path.join(git.get_src_root(),
                                             utils.get_build_platlib(),
                                             "benzina",
                                             "lib")],
                  runtime_library_dirs=[os.path.join("$ORIGIN", "lib")],
                  define_macros=[("PY_SSIZE_T_CLEAN", None)],
                  libraries=["benzina"],
        ),
    ],
    cmdclass={
        "build_configure": utils.build_configure,
        "build_ext":       utils.build_ext,
        "clean":           utils.clean,
        "meson_test":      utils.meson_test,
    },
    command_options={
        'build_sphinx':{
            'project':   ("setup.py", package_name),
            'copyright': ("setup.py", "2020, {}".format(author)),
            'version':   ("setup.py", versioning.ver_release),
            'release':   ("setup.py", versioning.ver_public)
        }
    },
)
