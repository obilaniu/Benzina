/* Include Guard */
#ifndef INCLUDE_BENZINA_CONFIG_H
#define INCLUDE_BENZINA_CONFIG_H


/**
 * @brief Compiler Interface and Identity
 * 
 * A compiler can present an interface not necessarily the same as its identity.
 * For instance, Clang can present a GCC or MVCS interface, so a mere check for
 * the GCC/MVSC macros is insufficient.
 * 
 * The interface macros are BENZINA_COMPILER_<X>_LIKE and the identity macros
 * (to the extent that identity can be accurately discerned) are BENZINA_COMPILER_<X>.
 */

#if    __GNUC__
# define BENZINA_COMPILER_GCC_LIKE                  1
#endif
#ifdef __clang__
# define BENZINA_COMPILER_CLANG_LIKE                1
#endif
#if    _MSC_VER
# define BENZINA_COMPILER_MSVC_LIKE                 1
#endif

#if !BENZINA_COMPILER_IDENTIFIED
# undef BENZINA_COMPILER_IDENTIFIED
# if   defined(__PGI)
   /* No known emulations of PGI except itself */
#  define BENZINA_COMPILER_PGI                      1
# elif defined(__INTEL_COMPILER)
   /* No known emulations of ICC except itself */
#  define BENZINA_COMPILER_INTEL                    1
# elif defined(__clang__)
  /* Clang can emulate GCC or MSVC, so must appear before both */
#  define BENZINA_COMPILER_CLANG                    1
# elif  __GNUC__
   /* GCC is the standard on Unix. */
#  define BENZINA_COMPILER_GCC                      1
# elif  _MSC_VER
   /* MSVC is the standard on Windows. */
#  define BENZINA_COMPILER_MSVC                     1
# else
#  define BENZINA_COMPILER_IDENTIFIED               0
# endif
# ifndef BENZINA_COMPILER_IDENTIFIED
#  define BENZINA_COMPILER_IDENTIFIED               1
# endif
#endif


/**
 * @brief Endianness
 */

#ifdef __BYTE_ORDER__
# if   __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define BENZINA_ENDIAN_BIG    0
#  define BENZINA_ENDIAN_LITTLE 1
# elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define BENZINA_ENDIAN_BIG    1
#  define BENZINA_ENDIAN_LITTLE 0
# else
#  define BENZINA_ENDIAN_BIG    0
#  define BENZINA_ENDIAN_LITTLE 0
# endif
#else
# define BENZINA_ENDIAN_BIG    @BENZINA_ENDIAN_BIG@
# define BENZINA_ENDIAN_LITTLE @BENZINA_ENDIAN_LITTLE@
#endif

#if !BENZINA_ENDIAN_BIG && !BENZINA_ENDIAN_LITTLE
# error "PDP-endian and Honeywell-endian machines are not supported!"
#endif


/**
 * @brief CPU Family
 */

#if !BENZINA_CPU_FAMILY_IDENTIFIED
# undef BENZINA_CPU_FAMILY_IDENTIFIED
# if   defined(__aarch64__)
#  define BENZINA_CPU_FAMILY_AARCH64                 1
# elif defined(__arm__) || defined(__thumb__) || defined(_M_ARM) || defined(_M_ARMT)
#  define BENZINA_CPU_FAMILY_ARM                     1
# elif defined(__alpha__) || defined(_M_ALPHA)
#  define BENZINA_CPU_FAMILY_ALPHA                   1
# elif defined(__amd64__) || defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64)
#  define BENZINA_CPU_FAMILY_X86_64                  1
# elif defined(__i386__) || defined(__i386) || defined(_M_IX86)
#  define BENZINA_CPU_FAMILY_X86                     1
# elif defined(__mips64__) || defined(__mips64)
#  define BENZINA_CPU_FAMILY_MIPS64                  1
# elif defined(__mips__) || defined(__mips)
#  define BENZINA_CPU_FAMILY_MIPS                    1
# elif defined(__powerpc64__) || defined(__ppc64__)
#  define BENZINA_CPU_FAMILY_PPC64                   1
# elif defined(__powerpc__) || defined(__ppc__) || defined(_M_PPC)
#  define BENZINA_CPU_FAMILY_PPC                     1
# else
#  define BENZINA_CPU_FAMILY_IDENTIFIED              0
# endif
# ifndef BENZINA_CPU_FAMILY_IDENTIFIED
#  define BENZINA_CPU_FAMILY_IDENTIFIED              1
# endif
#endif


/**
 * @brief Operating System
 */

#if !BENZINA_OS_IDENTIFIED
# undef BENZINA_OS_IDENTIFIED
# if   __APPLE__ && __MACH__
#  define BENZINA_OS_DARWIN                  1
# elif defined(__linux__)
#  define BENZINA_OS_LINUX                   1
# elif defined(__DragonFly__)
#  define BENZINA_OS_DRAGONFLY               1
# elif defined(__OpenBSD__)
#  define BENZINA_OS_OPENBSD                 1
# elif defined(__NetBSD__)
#  define BENZINA_OS_NETBSD                  1
# elif defined(__FreeBSD__)
#  define BENZINA_OS_FREEBSD                 1
# elif defined(_WIN32) || defined(_WIN64)
#  define BENZINA_OS_WINDOWS                 1
# else
#  define BENZINA_OS_IDENTIFIED              0
# endif
# ifndef BENZINA_OS_IDENTIFIED
#  define BENZINA_OS_IDENTIFIED              1
# endif
#endif


/* End Include Guard */
#endif

