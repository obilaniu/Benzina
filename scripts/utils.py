# -*- coding: utf-8 -*-
from . import git
import setuptools.command.build_ext
import distutils.command.clean
from   distutils.util      import get_platform
from   distutils.file_util import copy_file
from   distutils.dir_util  import copy_tree
import ast
import glob
import os, sys, shutil
import subprocess



def get_build_platlib():
    build_platlib = get_platform()
    build_platlib = ".%s-%d.%d" % (build_platlib, *sys.version_info[:2])
    if hasattr(sys, "gettotalrefcount"):
        build_platlib += "-pydebug"
    build_platlib = os.path.join("build", "lib"+build_platlib)
    return build_platlib

def get_meson_build_root(build_temp):
    meson_build_root = os.path.basename(build_temp)
    meson_build_root = os.path.join(os.path.dirname(build_temp),
                                    "meson"+meson_build_root[4:])
    return meson_build_root


class build_mixin:
    @property
    def build_meson(self):
        return get_meson_build_root(self.build_temp)


class build_configure(setuptools.command.build_ext.build_ext, build_mixin):
    description  = "Configure Meson build system."
    user_options = [
        ('reconfigure', 'r', 'Whether to forcibly reconfigure or not')
    ]
    
    def initialize_options(self):
        super().initialize_options()
        self.reconfigure = 0
    
    def run(self):
        absSrcRoot       = git.getSrcRoot()
        srcRoot          = os.path.relpath(absSrcRoot, self.build_meson)
        libRoot          = os.path.abspath(self.build_lib)
        
        os.makedirs(self.build_meson, exist_ok=True)
        
        if not os.path.isfile(os.path.join(self.build_meson,
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
                                  cwd   = self.build_meson)


class build_ext(setuptools.command.build_ext.build_ext, build_mixin):
    def run(self):
        self.get_finalized_command("build_configure").run()
        subprocess.check_call(["ninja"],
                              stdin  = subprocess.DEVNULL,
                              cwd    = self.build_meson)
        subprocess.check_call(["ninja", "install"],
                              stdin  = subprocess.DEVNULL,
                              cwd    = self.build_meson)
        super().run()
    
    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        
        #
        # Reference:
        #    https://github.com/pypa/setuptools/blob/211b194bee365b19aad10a487b20b48b17eb5c19/setuptools/command/build_ext.py#L83-L103
        #
        
        package_dir = self.get_finalized_command('build_py').get_package_dir("")
        for meson_out_file in self._get_meson_outputs():
            src_path = os.path.join(self.build_lib, meson_out_file)
            dst_path = os.path.join(package_dir,    meson_out_file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if os.path.islink(src_path):
                #
                # The function distutils.file_util.copy_file() cannot copy
                # symlinks. We invoke it anyways in unconditional dry-run mode,
                # for pretty-printing consistency, but use shutil.copy2() in
                # nofollow mode for the real action.
                #
                copy_file(src_path, dst_path,
                          verbose = self.verbose,
                          dry_run = True)
                if not self.dry_run:
                    if os.path.lexists(dst_path):
                        os.unlink(dst_path)
                    shutil.copy2(src_path, dst_path,
                                 follow_symlinks=False)
            else:
                copy_file(src_path, dst_path,
                          verbose = self.verbose,
                          dry_run = self.dry_run)
    
    def get_outputs(self):
        meson_targets = [
            os.path.abspath(os.path.join(self.build_lib, meson_out_file))
            for meson_out_file in self._get_meson_outputs()
        ]
        return super().get_outputs() + meson_targets
    
    def _get_meson_outputs(self):
        outs = subprocess.check_output(["meson", "introspect", "--installed"],
                                       universal_newlines = True,
                                       stdin  = subprocess.DEVNULL,
                                       cwd    = self.build_meson)
        outs = ast.literal_eval(outs)
        
        abs_dir       = os.path.abspath(self.build_lib)
        meson_targets = outs.values()
        meson_targets = [os.path.abspath(t)            for t in meson_targets]
        meson_targets = [os.path.relpath(t, abs_dir)   for t in meson_targets]
        
        
        ###
        # Required because Meson forgets to include .so and .so.x symlinks in
        # the output of `meson introspect --installed`. Hugely inefficient.
        ###
        meson_targets_real = [os.path.join(abs_dir, t) for t in meson_targets]
        meson_targets_real = [os.path.realpath(t)      for t in meson_targets_real]
        meson_targets_real_set = set(meson_targets_real)
        meson_extra_set        = set()
        meson_parent_dict      = {}
        for t,r in zip(meson_targets, meson_targets_real):
            dt = os.path.dirname(t)
            dr = os.path.dirname(r)
            if dt not in meson_parent_dict:
                meson_parent_dict[dt] = {
                    e for e in os.listdir(dr) if os.path.islink(os.path.join(dr,e))
                }
        meson_parent_dict = {k:v for k,v in meson_parent_dict.items() if v}
        for dt,s in meson_parent_dict.items():
            dr = os.path.realpath(os.path.join(abs_dir, dt))
            for e in s:
                f = os.path.join(dr,e)
                if f                   not in meson_targets_real_set and \
                   os.path.realpath(f)     in meson_targets_real_set:
                    meson_extra_set.add(os.path.join(dt,e))
        meson_targets.extend(sorted(list(meson_extra_set)))
        ###
        # Done attempting to determine missing symlinks.
        ###
        
        
        return meson_targets


class clean(distutils.command.clean.clean, build_mixin):
    def run(self):
        if os.path.exists(self.build_meson):
            distutils.dir_util.remove_tree(self.build_meson, dry_run=self.dry_run)
        
        return super().run()




