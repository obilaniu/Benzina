# -*- coding: utf-8 -*-
from . import git
import setuptools.command.build_ext
import distutils.command.clean
from   distutils.util      import get_platform
from   distutils.file_util import copy_file
from   distutils.dir_util  import copy_tree
import ast
import glob
import os, re, sys
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

class build_ext(setuptools.command.build_ext.build_ext):
	def run(self):
		mesonBuildRoot = get_meson_build_root(self.build_temp)
		absSrcRoot     = git.getSrcRoot()
		srcRoot        = os.path.relpath(absSrcRoot, mesonBuildRoot)
		libRoot        = os.path.abspath(self.build_lib)
		
		try:    os.mkdir(os.path.dirname(mesonBuildRoot))
		except: pass
		try:    os.mkdir(mesonBuildRoot)
		except: pass
		
		if not os.path.isfile(os.path.join(mesonBuildRoot,
		                                   "meson-private",
		                                   "coredata.dat")):
			subprocess.check_call(["meson",             srcRoot,
			                       "--prefix",          libRoot,
			                       "--buildtype",       "release",
			                       "-Dbuilding_py_pkg=true",
			                       "-Dcuda_runtime=static",
			                       "-Dpy_interpreter="+sys.executable,
			                       "-Dcuda_arch="+os.environ.get("CUDA_ARCH", "Auto"),
			                       "-Dcuda_home="+os.environ.get("CUDA_HOME", "/usr/local/cuda")],
			                      stdin  = subprocess.DEVNULL,
			                      cwd    = mesonBuildRoot)
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


def cuda_ver_cmp(a,b):
	aM, am = a.split(".")[:2]
	aM, am = int(aM), int(am)
	
	bM, bm = b.split(".")[:2]
	bM, bm = int(bM), int(bm)
	
	if aM-bM != 0:
		return aM-bM
	else:
		return am-bm


def cuda_detect_cuda_version(nvcc):
	out = subprocess.check_output([nvcc, '--version'],
	                              universal_newlines = True,
	                              stdin  = subprocess.DEVNULL,
	                              stderr = subprocess.PIPE)

	# nvcc was found in the PATH. We parse the version number from nvcc --version.
	#
	#     Example nvcc --version printout:
	#         nvcc: NVIDIA (R) Cuda compiler driver
	#         Copyright (c) 2005-2016 NVIDIA Corporation
	#         Built on Sun_Sep__4_22:14:01_CDT_2016
	#         Cuda compilation tools, release 8.0, V8.0.44
	#     We want                                   ^^^^^^
	if 'NVIDIA' in out:
		version = re.search("V\d+\.\d+\.\d+\n", out)
		if version:
			version = out[version.start()+1:version.end()-1]
		else:
			raise RuntimeError('Can\'t parse version number of "' + ' '.join(exelist) + '"')
	else:
		raise RuntimeError("Unknown compiler "+nvcc)
	return version


def cuda_detect_installed_gpus(nvcc, buildDir):
	"""Detect the compute capabilities of the installed GPUs."""
	
	testSrc = """
	#include <cuda_runtime.h>
	#include <stdio.h>
	
	int main(void){
		struct cudaDeviceProp prop;
		int count, i;
		if(cudaGetDeviceCount(&count) != cudaSuccess){return -1;}
		if(count == 0){return -1;}
		for(i=0;i<count;i++){
			if(cudaGetDeviceProperties(&prop, i) == cudaSuccess){
				printf("%d.%d ", prop.major, prop.minor);
			}
		}
		return 0;
	}
	"""
	
	outs      = ""
	cwd       = os.path.abspath(buildDir)
	pathToSrc = os.path.join(cwd, "./detect_cuda_compute_capabilities.c")
	pathToExe = os.path.join(cwd, "./detect_cuda_compute_capabilities")
	
	try:
		with open(pathToSrc, "w") as f:
			f.write(testSrc)
		
		subprocess.check_call([nvcc, pathToSrc, "--cudart", "static", "-o", pathToExe],
		                      stdin  = subprocess.DEVNULL,
		                      stdout = subprocess.DEVNULL,
		                      stderr = subprocess.DEVNULL,
		                      cwd    = cwd)
		
		outs = subprocess.check_output([pathToExe],
		                               universal_newlines = True,
		                               stdin  = subprocess.DEVNULL,
		                               cwd    = cwd)
	except: pass
	finally:
		try:    os.remove(pathToSrc)
		except: pass
		try:    os.remove(pathToExe)
		except: pass
	
	return outs


def cuda_select_nvcc_arch_flags(cuda_version, cuda_arch_list="Auto", detected=""):
	"""
	Using the CUDA version (the NVCC version) and the target architectures,
	compute the nvcc architecture flags.
	"""
	
	cuda_known_gpu_architectures  = ["Fermi", "Kepler", "Maxwell"]
	cuda_common_gpu_architectures = ["3.0", "3.5", "5.0"]
	cuda_limit_gpu_architecture   = None
	cuda_all_gpu_architectures    = ["3.0", "3.2", "3.5", "5.0"]
	
	if cuda_ver_cmp(cuda_version, "7.0")  < 0:
		cuda_limit_gpu_architecture = "5.2"
	
	if cuda_ver_cmp(cuda_version, "7.0") >= 0:
		cuda_known_gpu_architectures  += ["Kepler+Tegra", "Kepler+Tesla", "Maxwell+Tegra"]
		cuda_common_gpu_architectures += ["5.2"]
		
		if cuda_ver_cmp(cuda_version, "8.0") < 0:
			cuda_common_gpu_architectures += ["5.2+PTX"]
			cuda_limit_gpu_architecture    = "6.0"
	
	if cuda_ver_cmp(cuda_version, "8.0") >= 0:
		cuda_known_gpu_architectures  += ["Pascal"]
		cuda_common_gpu_architectures += ["6.0", "6.1"]
		cuda_all_gpu_architectures    += ["6.0", "6.1", "6.2"]
		
		if cuda_ver_cmp(cuda_version, "9.0") < 0:
			cuda_common_gpu_architectures += ["6.1+PTX"]
			cuda_limit_gpu_architecture    = "7.0"
	
	if cuda_ver_cmp(cuda_version, "9.0") >= 0:
		cuda_known_gpu_architectures  += ["Volta"]
		cuda_common_gpu_architectures += ["7.0", "7.0+PTX"]
		
		if cuda_ver_cmp(cuda_version, "10.0") < 0:
			cuda_limit_gpu_architecture    = "8.0"
	
	
	if not cuda_arch_list:
		cuda_arch_list = "Auto"
	
	if   cuda_arch_list == "All":
		cuda_arch_list = cuda_known_gpu_architectures
	elif cuda_arch_list == "Common":
		cuda_arch_list = cuda_common_gpu_architectures
	elif cuda_arch_list == "Auto":
		if detected:
			if isinstance(detected, list):
				cuda_arch_list = detected
			else:
				cuda_arch_list = re.sub("[ \t]+", ";", detected).split(";")
			
			if cuda_limit_gpu_architecture:
				filtered_cuda_arch_list = []
				for arch in cuda_arch_list:
					if arch:
						if cuda_ver_cmp(arch, cuda_limit_gpu_architecture) >= 0:
							filtered_cuda_arch_list.append(cuda_common_gpu_architectures[-1])
						else:
							filtered_cuda_arch_list.append(arch)
				cuda_arch_list = filtered_cuda_arch_list
		else:
			cuda_arch_list = cuda_common_gpu_architectures
	elif isinstance(cuda_arch_list, str):
		cuda_arch_list = re.sub("[ \t]+", ";", cuda_arch_list).split(";")
	
	cuda_arch_list = sorted([x for x in set(cuda_arch_list) if x])
	
	cuda_arch_bin = []
	cuda_arch_ptx = []
	for arch_name in cuda_arch_list:
		arch_bin = []
		arch_ptx = []
		add_ptx  = False
		
		if arch_name.endswith("+PTX"):
			add_ptx   = True
			arch_name = arch_name[:-len("+PTX")]
		
		if re.fullmatch("""[0-9]+\.[0-9](\([0-9]+\.[0-9]\))?""", arch_name):
			arch_bin = [arch_name]
			arch_ptx = [arch_name]
		else:
			if   arch_name == "Fermi":         arch_bin=["2.0", "2.1(2.0)"]
			elif arch_name == "Kepler+Tegra":  arch_bin=["3.2"]
			elif arch_name == "Kepler+Tesla":  arch_bin=["3.7"]
			elif arch_name == "Kepler":        arch_bin=["3.0", "3.5"]; arch_ptx=["3.5"]
			elif arch_name == "Maxwell+Tegra": arch_bin=["5.3"]
			elif arch_name == "Maxwell":       arch_bin=["5.0", "5.2"]; arch_ptx=["5.2"]
			elif arch_name == "Pascal":        arch_bin=["6.0", "6.1"]; arch_ptx=["6.1"]
			elif arch_name == "Pascal":        arch_bin=["7.0", "7.0"]; arch_ptx=["7.0"]
			else: raise ValueError("Unknown CUDA Architecture Name "+arch_name+
			                       " in cuda_select_nvcc_arch_flags()!")
		
		if not arch_bin:
			raise ValueError("arch_bin wasn't set for some reason")
		
		cuda_arch_bin += arch_bin
		
		if add_ptx:
			if not arch_ptx:
				arch_ptx = arch_bin
			cuda_arch_ptx += arch_ptx
			
	cuda_arch_bin = re.sub    ("\.",  "",  " ".join(cuda_arch_bin))
	cuda_arch_ptx = re.sub    ("\.",  "",  " ".join(cuda_arch_ptx))
	cuda_arch_bin = re.findall("[0-9()]+", cuda_arch_bin)
	cuda_arch_ptx = re.findall("[0-9]+",   cuda_arch_ptx)
	
	if cuda_arch_bin: cuda_arch_bin = sorted(list(set(cuda_arch_bin)))
	if cuda_arch_ptx: cuda_arch_ptx = sorted(list(set(cuda_arch_ptx)))
	
	nvcc_flags          = []
	nvcc_archs_readable = []
	
	for arch in cuda_arch_bin:
		m = re.match("""([0-9]+)\(([0-9]+)\)""", arch)
		if m:
			nvcc_flags          += ["-gencode", "arch=compute_{},code=sm_{}".format(m[1], m[0])]
			nvcc_archs_readable += ["sm_"+m[0]]
		else:
			nvcc_flags          += ["-gencode", "arch=compute_"+arch+",code=sm_"+arch]
			nvcc_archs_readable += ["sm_"+arch]
	
	for arch in cuda_arch_ptx:
		nvcc_flags          += ["-gencode", "arch=compute_"+arch+",code=compute_"+arch]
		nvcc_archs_readable += ["compute_"+arch]
	
	return nvcc_flags, nvcc_archs_readable

