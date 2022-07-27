from . import git
import setuptools.command.build_ext
import distutils.command.clean
import distutils.command.build
from   distutils.util      import get_platform
from   distutils.file_util import copy_file
from   distutils.dir_util  import copy_tree
import ast
import glob
import os, shlex, sys, shutil
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
        """
        Perform Meson (re)configuration step, if it hasn't been done already.
        
        Meson does not allow manual access to the environment variables from
        inside meson.build files, preferring Meson options. It is also
        needlessly noisy when PKG_CONFIG_PATH is set as an environment
        variable and/or contains duplicates, so we do what it prefers: We pop
        out PKG_CONFIG_PATH from the environment, strip duplicates ourselves,
        then use Meson's -Dpkg_config_path=string option.
        """
        
        os.makedirs(self.build_meson, exist_ok=True)
        if os.path.isfile(os.path.join(self.build_meson,
                                       "meson-private",
                                       "coredata.dat")) and not self.reconfigure:
            return None
        
        
        #
        # Custom environment for meson configure step.
        #   - PYTHONUNBUFFERED: Prompt display of messages during configure
        #   - GIT_CONFIG_{COUNT|KEY|VALUE}: Silence polluting detached-head advice.
        #   - PKG_CONFIG_PATH:  Silence warning about duplicates in env or option.
        #
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if "GIT_CONFIG_COUNT" not in env:
            env["GIT_CONFIG_COUNT"]   = "1"
            env["GIT_CONFIG_KEY_0"]   = "advice.detachedHead"
            env["GIT_CONFIG_VALUE_0"] = "false"
        
        pkg_config_path_dup = env.pop("PKG_CONFIG_PATH", "").split(":")
        pkg_config_path = []
        for p in pkg_config_path_dup:
            if p not in pkg_config_path:
                pkg_config_path.append(p)
        pkg_config_path = ":".join(pkg_config_path)
        
        extra_args = shlex.split(env.pop("MESON_ARGS", ""))
        
        cmd  = [
            "meson",            git.get_src_root(),
            "--prefix",         os.path.abspath(self.build_lib),
            "-Dbuilding_py_pkg="+"true",
            "-Dpy_interpreter=" +sys.executable,
            "-Dpkg_config_path="+pkg_config_path,
            "-Dbuildtype="      +env.get("BUILD_TYPE",    "release"),
            "-Denable_gpl="     +env.get("ENABLE_GPL",    "false"),
            "-Dnvidia_driver="  +env.get("NVIDIA_DRIVER", "Auto"),
            "-Dnvidia_runtime=" +env.get("CUDA_RUNTIME",  "static"),
            "-Dnvidia_arch="    +env.get("CUDA_ARCH",     "Auto"),
            "-Dnvidia_home="    +os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        ] + extra_args
        if self.reconfigure: cmd.append("--reconfigure")
        
        subprocess.check_call(cmd,
                              stdin = subprocess.DEVNULL,
                              cwd   = self.build_meson,
                              env   = env)


class build_ext(setuptools.command.build_ext.build_ext, build_mixin):
    def run(self):
        self.run_command("build_configure")
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
        return meson_targets


class clean(distutils.command.clean.clean, build_mixin):
    def run(self):
        if os.path.exists(self.build_meson):
            distutils.dir_util.remove_tree(self.build_meson, dry_run=self.dry_run)
        
        return super().run()


class meson_test(distutils.command.build.build, build_mixin):
    description  = "Run Meson tests."
    
    def run(self):
        self.run_command('build_configure')
        subprocess.check_call(["meson", "test"],
                              stdin = subprocess.DEVNULL,
                              cwd   = self.build_meson)


