#/bin/sh
LUA_FORK="${1:-${LUA_FORK:?"Lua fork not specified!"}}"
LUA_STARTPATCH="${LUA_STARTPATCH:-":/Enable renaming main functions"}"
LUA_STARTPATCH="$(git -C "$LUA_FORK" rev-parse "$LUA_STARTPATCH")"
LUA_SUBPROJECT="$(pwd)/subprojects/packagefiles/lua"
rm -fv "$LUA_SUBPROJECT/patches"/????-*.patch
touch  "$LUA_SUBPROJECT/patches/9999-Empty.patch"
git -C "$LUA_FORK" format-patch  \
    -o "$LUA_SUBPROJECT/patches" \
    --filename-max-length=64     \
    --zero-commit -N             \
    --no-signature               \
    --no-stat                    \
    --binary                     \
    "${LUA_STARTPATCH}^.."
