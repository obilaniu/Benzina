ASM_SECT_LICENSE_DECL
    .incbin "../../LICENSE.md"
    .ascii  "\n\n"

#if BENZINA_OS_DARWIN
#
# On Darwin-based systems, we cannot make use of linker scripts because the
# system linker, ld64, doesn't support them.
#
# We declare here certain "magic" symbols and make them global.
# Then, we define them in terms of section start and end addresses,
# synthesized by the linker.
#

.globl _license_benzina

.globl ___lz4_cmd_array_start
.globl ___lz4_cmd_array_end

.globl ___tool_array_start
.globl ___tool_array_end

.globl ___lua_open_array_start
.globl ___lua_open_array_end

.globl ___lua_text_array_start
.globl ___lua_text_array_end


.set   _license_benzina,                section$start$__TEXT$__license

.set   ___lz4_cmd_array_start,          section$start$__DATA$__lz4_cmd_array
.set   ___lz4_cmd_array_end,            section$end$__DATA$__lz4_cmd_array

.set   ___tool_array_start,             section$start$__DATA$__tool_array
.set   ___tool_array_end,               section$end$__DATA$__tool_array

.set   ___lua_open_array_start,         section$start$__DATA$__lua_open_array
.set   ___lua_open_array_end,           section$end$__DATA$__lua_open_array

.set   ___lua_text_array_start,         section$start$__DATA$__lua_text_array
.set   ___lua_text_array_end,           section$end$__DATA$__lua_text_array
#endif
