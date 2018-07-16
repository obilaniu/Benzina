import os, sys
sys.path.insert(0, os.environ.get("MESON_SOURCE_ROOT"))
from scripts.utils import (cuda_detect_cuda_version,
                           cuda_detect_installed_gpus,
                           cuda_select_nvcc_arch_flags)


if __name__ == "__main__":
	nvcc, cuda_arch_list = sys.argv[1], sys.argv[2]
	cuda_version = cuda_detect_cuda_version(nvcc)
	detected = ""
	if cuda_arch_list == "Auto":
		detected = cuda_detect_installed_gpus(nvcc, os.environ.get("MESON_BUILD_ROOT"))
	
	flags, readable = cuda_select_nvcc_arch_flags(cuda_version, cuda_arch_list, detected)
	
	sys.stdout.write(os.linesep.join(flags))
	if cuda_arch_list == "Auto":
		if detected:
			sys.stderr.write("Building for detected GPUs [{}]".format(" ".join(readable)))
		else:
			sys.stderr.write("Building for common GPUs [{}]".format(" ".join(readable)))
	else:
		sys.stderr.write("Building for selected GPUs [{}]".format(" ".join(readable)))
