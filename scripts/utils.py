# -*- coding: utf-8 -*-
from . import git
import setuptools.command.build_ext
import distutils.command.clean
from   distutils.util      import get_platform
from   distutils.file_util import copy_file
from   distutils.dir_util  import copy_tree
import ast
import glob
import os, sys
import subprocess



def get_build_platlib():
    build_platlib = get_platform()
    build_platlib = ".%s-%d.%d" % (build_platlib, *sys.version_info[:2])
    if hasattr(sys, "gettotalrefcount"):
        build_platlib += "-pydebug"
    build_platlib = os.path.join("build", "lib"+build_platlib)
    return build_platlib

def get_meson_build_root(build_temp):
    mesonBuildRoot = os.path.basename(build_temp)
    mesonBuildRoot = os.path.join(os.path.dirname(build_temp),
                                  "meson"+mesonBuildRoot[4:])
    return mesonBuildRoot


class build_configure(setuptools.command.build_ext.build_ext):
    description  = "Configure Meson build system."
    user_options = [
        ('reconfigure', 'r', 'Whether to forcibly reconfigure or not')
    ]
    
    def initialize_options(self):
        super().initialize_options()
        self.reconfigure = 0
    
    def run(self):
        mesonBuildRoot = get_meson_build_root(self.build_temp)
        absSrcRoot     = git.getSrcRoot()
        srcRoot        = os.path.relpath(absSrcRoot, mesonBuildRoot)
        libRoot        = os.path.abspath(self.build_lib)
        
        os.makedirs(mesonBuildRoot, exist_ok=True)
        
        if not os.path.isfile(os.path.join(mesonBuildRoot,
                                           "meson-private",
                                           "coredata.dat")) or self.reconfigure:
            cmd  = ["meson",            srcRoot,
                    "--prefix",         libRoot,
                    "-Dbuildtype="      +os.environ.get("BUILD_TYPE", "release"),
                    "-Dbuilding_py_pkg="+"true",
                    "-Denable_gpl="     +os.environ.get("ENABLE_GPL", "false"),
                    "-Dpy_interpreter=" +sys.executable,
                    "-Dnvidia_runtime=" +os.environ.get("CUDA_RUNTIME", "static"),
                    "-Dnvidia_arch="    +os.environ.get("CUDA_ARCH", "Auto"),
                    "-Dnvidia_home="    +os.environ.get("CUDA_HOME", "/usr/local/cuda"),
                    "-Dnvidia_driver="  +os.environ.get("NVIDIA_DRIVER", "Auto")]
            if self.reconfigure: cmd.append("--reconfigure")
            subprocess.check_call(cmd,
                                  stdin = subprocess.DEVNULL,
                                  cwd   = mesonBuildRoot)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.get_finalized_command("build_configure").run()
        
        mesonBuildRoot = get_meson_build_root(self.build_temp)
        subprocess.check_call(["ninja"],
                              stdin  = subprocess.DEVNULL,
                              cwd    = mesonBuildRoot)
        subprocess.check_call(["ninja", "install"],
                              stdin  = subprocess.DEVNULL,
                              cwd    = mesonBuildRoot)
        
        super().run()
    
    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        
        #
        # Reference:
        #    https://github.com/pypa/setuptools/blob/211b194bee365b19aad10a487b20b48b17eb5c19/setuptools/command/build_ext.py#L83-L103
        #
        
        build_py    = self.get_finalized_command('build_py')
        package_dir = build_py.get_package_dir("")
        
        mesonBuildRoot = get_meson_build_root(self.build_temp)
        outs = subprocess.check_output(["meson", "introspect", "--installed"],
                                       universal_newlines = True,
                                       stdin  = subprocess.DEVNULL,
                                       cwd    = mesonBuildRoot)
        outs = ast.literal_eval(outs)
        
        for tmpFile, stagedFile in outs.items():
            stagedFile = os.path.relpath(os.path.abspath(stagedFile),
                                         os.path.abspath(self.build_lib))
            srcPath    = os.path.join(self.build_lib, stagedFile)
            dstPath    = os.path.join(package_dir,    stagedFile)
            
            os.makedirs(os.path.dirname(dstPath), exist_ok=True)
            
            copy_file(srcPath, dstPath,
                      verbose = self.verbose,
                      dry_run = self.dry_run)
    
    def get_outputs(self):
        build_py       = self.get_finalized_command('build_py')
        package_dir    = build_py.get_package_dir("")
        mesonBuildRoot = get_meson_build_root(self.build_temp)
        outs = subprocess.check_output(["meson", "introspect", "--installed"],
                                       universal_newlines = True,
                                       stdin  = subprocess.DEVNULL,
                                       cwd    = mesonBuildRoot)
        outs = ast.literal_eval(outs)
        
        mesonLibs = []
        for tmpFile, stagedFile in outs.items():
            stagedFile = os.path.relpath(os.path.abspath(stagedFile),
                                         os.path.abspath(self.build_lib))
            stagedFile = os.path.abspath(os.path.join(package_dir, stagedFile))
            mesonLibs.append(stagedFile)
        
        return super().get_outputs() + mesonLibs


class clean(distutils.command.clean.clean):
    def run(self):
        #
        # We invoke Meson before running the rest.
        #
        
        mesonBuildRoot = get_meson_build_root(self.build_temp)
        if os.path.exists(mesonBuildRoot):
            distutils.dir_util.remove_tree(mesonBuildRoot, dry_run=self.dry_run)
        
        return super().run()




