#
# Lua's source code must be patched, but Meson does not make it entirely easy
# to do so. Meson does support *overlaying* files atop the source distribution,
# but this requires a copy of the entire file, even for a single-line change.
#
# We contort ourselves into a pretzel here to patch Lua's source code using
# patchfiles.
#

core_filenames = [
    'lapi.c',    'lcode.c',
    'lctype.c',  'ldebug.c',
    'ldo.c',     'ldump.c',
    'lfunc.c',   'lgc.c',
    'llex.c',    'lmem.c',
    'lobject.c', 'lopcodes.c',
    'lparser.c', 'lstate.c',
    'lstring.c', 'ltable.c',
    'ltm.c',     'lundump.c',
    'lvm.c',     'lzio.c',
]
aux_filenames = ['lauxlib.c']
lib_filenames = [
   'lbaselib.c', 'lcorolib.c',
   'ldblib.c',   'liolib.c',
   'lmathlib.c', 'loadlib.c',
   'loslib.c',   'lstrlib.c',
   'ltablib.c',  'lutf8lib.c',
   'linit.c',
]
hdr_filenames = [
    'lapi.h',     'lauxlib.h',
    'lcode.h',    'lctype.h',
    'ldebug.h',   'ldo.h',
    'lfunc.h',    'lgc.h',
    'ljumptab.h', 'llex.h',
    'llimits.h',  'lmem.h',
    'lobject.h',  'lopcodes.h',
    'lopnames.h', 'lparser.h',
    'lprefix.h',  'lstate.h',
    'lstring.h',  'ltable.h',
    'ltm.h',      'luaconf.h',
    'lua.h',      'lua.hpp',
    'lualib.h',   'lundump.h',
    'lvm.h',      'lzio.h',
    # Custom:
    'luabenz.h',
]
lua_filenames  = ['lua.c']
luac_filenames = ['luac.c']
patch_filenames = [
    '000-lua-5.4.3-postrel.patch',     # Update from 5.4.3 with postrelease bugfixes
    '001-main-function-rename.patch',  # Allow renaming main()
    '002-luaopen-extra-preload.patch', # Allow preloading of statically-linked libraries.
    '003-searcher-static-C.patch',     # Allow demand-loading of statically-linked libraries.
]
lua_inc = include_directories('.')


all_patched_files = custom_target(
    'Lua patched files',
    command: [files('apply-patches.sh'), '@CURRENT_SOURCE_DIR@', '@OUTDIR@', '@INPUT@'],
    input:   files(core_filenames+aux_filenames+lib_filenames+
                    hdr_filenames+lua_filenames+luac_filenames),
    output:       [core_filenames+aux_filenames+lib_filenames+
                    hdr_filenames+lua_filenames+luac_filenames],
    depend_files: files(patch_filenames),
    install: false,
)


i = 0
core_files = []
foreach f: core_filenames
  core_files += [all_patched_files[i]]
  i          += 1
endforeach
aux_files  = []
foreach f: aux_filenames
  aux_files  += [all_patched_files[i]]
  i          += 1
endforeach
lib_files  = []
foreach f: lib_filenames
  lib_files  += [all_patched_files[i]]
  i          += 1
endforeach
hdr_files  = []
foreach f: hdr_filenames
  hdr_files  += [all_patched_files[i]]
  i          += 1
endforeach
lua_files  = []
foreach f: lua_filenames
  lua_files  += [all_patched_files[i]]
  i          += 1
endforeach
luac_files = []
foreach f: luac_filenames
  luac_files += [all_patched_files[i]]
  i          += 1
endforeach

