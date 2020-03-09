#
# Imports
#
import os, subprocess


# Useful constants
EMPTYTREE_SHA1 = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ORIGINAL_ENV   = os.environ.copy()
C_ENV          = os.environ.copy()
C_ENV['LANGUAGE'] = C_ENV['LANG'] = C_ENV['LC_ALL'] = "C"
SCRIPT_PATH    = os.path.abspath(os.path.dirname(__file__))
SRCROOT_PATH   = None
GIT_VER        = None
GIT_CLEAN      = None


#
# Utility functions
#
def invoke(command,
           cwd    = SCRIPT_PATH,
           env    = C_ENV,
           stdin  = subprocess.DEVNULL,
           stdout = subprocess.PIPE,
           stderr = subprocess.PIPE,
           **kwargs):
    return subprocess.Popen(
        command,
        stdin  = stdin,
        stdout = stdout,
        stderr = stderr,
        cwd    = cwd,
        env    = env,
        **kwargs
    )

def get_src_root():
    #
    # Return the cached value if we know it.
    #
    global SRCROOT_PATH
    if SRCROOT_PATH is not None:
        return SRCROOT_PATH
    
    #
    # Our initial guess is `dirname(dirname(__file__))`.
    #
    root = os.path.dirname(SCRIPT_PATH)
    
    try:
        inv = invoke(["git", "rev-parse", "--show-toplevel"],
                     universal_newlines = True,)
        streamOut, streamErr = inv.communicate()
        if inv.returncode == 0:
            root = streamOut[:-1]
    except FileNotFoundError as err:
        pass
    finally:
        SRCROOT_PATH = root
    
    return root

def get_git_ver():
    #
    # Return the cached value if we know it.
    #
    global GIT_VER
    if GIT_VER is not None:
        return GIT_VER
    
    try:
        gitVer = ""
        inv    = invoke(["git", "rev-parse", "HEAD"],
                        universal_newlines = True,)
        streamOut, streamErr = inv.communicate()
        if inv.returncode == 0 or inv.returncode == 128:
            gitVer = streamOut[:-1]
    except FileNotFoundError as err:
        pass
    finally:
        if gitVer == "HEAD":
            GIT_VER = EMPTYTREE_SHA1
        else:
            GIT_VER = gitVer
    
    return GIT_VER

def is_git_clean():
    #
    # Return the cached value if we know it.
    #
    global GIT_CLEAN
    if GIT_CLEAN is not None:
        return GIT_CLEAN
    
    try:
        gitVer = None
        inv_nc = invoke(["git", "diff", "--quiet"],
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL,)
        inv_c  = invoke(["git", "diff", "--quiet", "--cached"],
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL,)
        inv_nc = inv_nc.wait()
        inv_c  = inv_c .wait()
        GIT_CLEAN = (inv_nc == 0) and (inv_c == 0)
    except FileNotFoundError as err:
        #
        # If we don't have access to Git, assume it's a tarball, in which case
        # it's always clean.
        #
        GIT_CLEAN = True
    
    return GIT_CLEAN
