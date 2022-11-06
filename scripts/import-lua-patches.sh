#/bin/sh -e
# Ensure we are invoked from root and that the text files appear to be sane.
test -f meson_options.txt
test -f subprojects/lua.wrap
grep -q '^diff_files ='     subprojects/lua.wrap
grep -q '9999-Empty.patch$' subprojects/lua.wrap


# Ensure a Lua fork has been specified, and set variables we need.
LUA_FORK="${1:-${LUA_FORK:?"Lua fork not specified!"}}"
LUA_STARTPATCH="${LUA_STARTPATCH:-":/Enable renaming main functions"}"
LUA_STARTPATCH="$(git -C "$LUA_FORK" rev-parse "$LUA_STARTPATCH")"
LUA_SUBPROJECT="$(pwd)/subprojects/packagefiles/lua"


# Remove old set of patches and regenerate with git format-patches.
mkdir -p "$LUA_SUBPROJECT/patches"
rm -f  "$LUA_SUBPROJECT/patches"/????-*.patch
touch  "$LUA_SUBPROJECT/patches/9999-Empty.patch"
git -C "$LUA_FORK" format-patch  \
    -o "$LUA_SUBPROJECT/patches" \
    --filename-max-length=64     \
    --zero-commit -N             \
    --no-signature               \
    --no-stat                    \
    --binary                     \
    --quiet                      \
    "${LUA_STARTPATCH}^.."


#
# Patches have been generated.
# Auto-edit w/ GNU sed their enumeration in lua.wrap:diff_files
# The enumeration in lua.wrap *must* resemble:
#
#     diff_files =
#         [...].patch,
#         [...].patch,
#         [...]9999-Empty.patch
#
# We make the update conditional upon there being an actual change,
# as the timestamp update for lua.wrap can become quite irritating.
#
LUA_PATCHLIST="diff_files =\\n$(
    cd "$LUA_SUBPROJECT/patches"
    ls ????-*.patch | sed -e '/9999-Empty.patch$/! {s|.*|    lua/patches/&,\\|g}' \
                          -e '/9999-Empty.patch$/  {s|.*|    lua/patches/&|;q}'
)"
if ! sed -e '/^diff_files =/,/9999-Empty.patch$/c\'    \
         -e "$LUA_PATCHLIST"    subprojects/lua.wrap | \
     diff -q - subprojects/lua.wrap > /dev/null; then
     sed -e '/^diff_files =/,/9999-Empty.patch$/c\'    \
         -e "$LUA_PATCHLIST" -i subprojects/lua.wrap
fi
