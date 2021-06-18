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
 * .section .lua.open_array, "aw"
 *     - A concatenation of entries
 *           struct luaL_Reg{const char* name; lua_CFunction func;};
 *       somewhat modelled on .init_array.
 *     - Array start: luaL_Reg __lua_open_array_start[]
 *     - Array end:   luaL_Reg __lua_open_array_end[]
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
 *       corresponding embedded source code or byte code and points into the
 *       .lua.text section.
 *     - This array is searched as one possible source for require'd packages.
 *
 * .section .lua.text, "a"
 *     - A concatenation or Lua source code or byte code with adequate symbols.
 *     - The pointers in the array of section .lua.text_array point here.
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
    .lua.text : ALIGN(4) {
        FILL(0x00000000);
        KEEP(*(.lua.text .lua.text.*));
        /**
         * If the final file is Lua source code, terminate it with a convenient
         * NUL byte or four, as though it were a C string. If it is Lua bytecode,
         * should maintain alignment.
         */
        QUAD(0);
    }
} INSERT AFTER .rodata;

SECTIONS {
    .lua.open_array : ALIGN(16) {
        HIDDEN(__lua_open_array_start = .);
        KEEP(*(.lua.open_array));
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
