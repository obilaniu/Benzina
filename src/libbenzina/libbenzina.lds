/**
 * libbenzina.so Linker Script.
 *
 * The library has several additional ELF sections over and beyond those of a
 * typical library. We add:
 *
 * .section .license, "a"
 *     - A concatenation of the LICENSE files of statically-linked subprojects.
 *     - First license text is that of Benzina itself, then any dependencies
 *       (in an unspecified order).
 *     - Defines symbol license_benzina that can be understood as const char[].
 *     - Linker script terminates string concatenation with a NUL byte to make
 *       the symbol above usable as a C string.
 *
 * .section .lz4.cmd_array, "aw"
 *     - A concatenation of entries
 *           struct lz4_entry{const char* src_start, *src_end;
 *                            char*       dst_start, *dst_end;};
 *       somewhat modelled on .init_array.
 *     - Array start:         lz4_entry __lz4_cmd_array_start[]
 *     - Array end:           lz4_entry __lz4_cmd_array_end[]
 *     - Every entry represents one decompression order.
 *
 * .section .tool_array, "aw"
 *     - A concatenation of entries
 *           struct entry{const char* name; int(*func)(int argc, char* argv[]);};
 *     - Array start: entry __tool_array_start[]
 *     - Array end:   entry __tool_array_end[]
 *     - Every <name> is a command/tool name. Every <func> is its corresponding
 *       main() function entry point.
 *
 * .section .lua.open_array, "aw"
 *     - A concatenation of entries
 *           struct luaL_Reg{const char* name; lua_CFunction func;};
 *       somewhat modelled on .init_array.
 *     - Array start:         luaL_Reg __lua_open_array_start[]
 *     - Array end:           luaL_Reg __lua_open_array_end[]
 *     - Every <name> is a Lua package name. Every <func> is its corresponding
 *       luaopen_<name>() initializer.
 *     - This array is searched as one possible source for require'd packages.
 *
 * .section .lua.text_array, "aw"
 *     - A concatenation of entries
 *           struct entry{const char* name; const char* start; const char* end;};
 *     - Array start: entry __lua_text_array_start[]
 *     - Array end:   entry __lua_text_array_end[]
 *     - Every <name> is a Lua package name. Every <start> and <end> is its
 *       corresponding embedded source code or byte code (May first require
 *       decompression in the library initializer!)
 *     - This array is searched as one possible source for require'd packages.
 */

SECTIONS {
    .license : ALIGN(1) SUBALIGN(1) {
        HIDDEN(license_benzina = .);
        /* TODO: Force first object file selected to be Benzina's license! */
        KEEP(*internal*.o(.license));
        ASSERT(. > license_benzina, "Failed to statically link Benzina LICENSE.md file!");
        KEEP(*(.license));
        ASSERT(. > license_benzina, "Failed to statically link LICENSE* files!");
        BYTE(0);
    }
} INSERT BEFORE .rodata;

SECTIONS {
    /**
     * This section is entirely optional, but the idea is that LZ4 data is
     * "cold" and so should be grouped together and behind the main .rodata
     * section.
     */
    .rodata.lz4 : {
        FILL(0x00000000);
        KEEP(*(.rodata.lz4 .rodata.lz4.*));
        QUAD(0); QUAD(0); /* Allow wild 8-byte overread. */
    }
} INSERT AFTER .rodata;

SECTIONS {
    .lz4.cmd_array : ALIGN(32) {
        HIDDEN(__lz4_cmd_array_start = .);
        KEEP(*(.lz4.cmd_array .lz4.cmd_array.*));
        HIDDEN(__lz4_cmd_array_end = .);
        QUAD(0); QUAD(0); QUAD(0); QUAD(0); /* NULL sentinel */
    }
    .tool_array : ALIGN(16) {
        HIDDEN(__tool_array_start = .);
        KEEP(*main*.o(.tool_array));
        ASSERT(. > __tool_array_start, "Failed to register Lua/Luac tools!");
        KEEP(*(.tool_array));
        HIDDEN(__tool_array_end = .);
        QUAD(0); QUAD(0);          /* NULL sentinel */
    }
    .lua.open_array : ALIGN(16) {
        HIDDEN(__lua_open_array_start = .);
        KEEP(*(.lua.open_array .lua.open_array.*));
        HIDDEN(__lua_open_array_end = .);
        QUAD(0); QUAD(0);          /* NULL sentinel */
    }
    .lua.text_array : ALIGN(8) {
        HIDDEN(__lua_text_array_start = .);
        KEEP(*(.lua.text_array));
        HIDDEN(__lua_text_array_end = .);
        QUAD(0); QUAD(0); QUAD(0); /* NULL sentinel */
    }
} INSERT BEFORE .fini_array;
