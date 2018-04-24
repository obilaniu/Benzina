# -*- coding: utf-8 -*-
from . import git
import setuptools.command.build_ext
import distutils.command.clean
from   distutils.util     import get_platform
from   distutils.dir_util import copy_tree
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


class build_ext(setuptools.command.build_ext.build_ext):
	def run(self):
		mesonBuildRoot = os.path.basename(self.build_temp)
		mesonBuildRoot = os.path.join(os.path.dirname(self.build_temp),
		                              "meson"+mesonBuildRoot[4:])
		absSrcRoot     = git.getSrcRoot()
		srcRoot        = os.path.relpath(absSrcRoot, mesonBuildRoot)
		libRoot        = os.path.abspath(self.build_lib)
		mesonBuildEnv  = os.environ.copy()
		mesonBuildEnv["SETUPTOOLS_DRIVING_MESON"] = "1"
		
		try:    os.mkdir(os.path.dirname(mesonBuildRoot))
		except: pass
		try:    os.mkdir(mesonBuildRoot)
		except: pass
		
		if not os.path.isfile(os.path.join(mesonBuildRoot,
		                                   "meson-private",
		                                   "coredata.dat")):
			subprocess.check_call(["meson", srcRoot, "--prefix", libRoot],
			                      stdin  = subprocess.DEVNULL,
			                      cwd    = mesonBuildRoot,
			                      env    = mesonBuildEnv)
		subprocess.check_call(["ninja"],
		                      stdin  = subprocess.DEVNULL,
		                      cwd    = mesonBuildRoot,
		                      env    = mesonBuildEnv)
		subprocess.check_call(["ninja", "install"],
		                      stdin  = subprocess.DEVNULL,
		                      cwd    = mesonBuildRoot,
		                      env    = mesonBuildEnv)
		
		super().run()
	
	def copy_extensions_to_source(self):
		super().copy_extensions_to_source()
		
		#
		# Reference:
		#    https://github.com/pypa/setuptools/blob/211b194bee365b19aad10a487b20b48b17eb5c19/setuptools/command/build_ext.py#L83-L103
		#
		
		build_py    = self.get_finalized_command('build_py')
		package_dir = build_py.get_package_dir("benzina")
		copy_tree(os.path.join(self.build_lib, "benzina", "libs"),
		          os.path.join(package_dir,               "libs"),
		          preserve_symlinks = True,
		          verbose           = self.verbose,
		          dry_run           = self.dry_run)
	
	def get_outputs(self):
		mesonLibs = glob.glob(os.path.join(self.build_lib,
		                                   "benzina",
		                                   "libs",
		                                   "**",
		                                   "*"),
		                      recursive=True)
		
		return super().get_outputs() + mesonLibs


class clean(distutils.command.clean.clean):
	def run(self):
		#
		# We invoke Meson before running the rest.
		#
		
		mesonBuildRoot = os.path.basename(self.build_temp)
		mesonBuildRoot = os.path.join(os.path.dirname(self.build_temp),
		                              "meson"+mesonBuildRoot[4:])
		
		if os.path.exists(mesonBuildRoot):
			distutils.dir_util.remove_tree(mesonBuildRoot, dry_run=self.dry_run)
		
		return super().run()
