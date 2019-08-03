/* Include Guards */
#ifndef INCLUDE_BENZINA_INLINE_H
#define INCLUDE_BENZINA_INLINE_H


/**
 * @brief Portable "inline" keyword.
 *
 * The situation around the keywords "inline", "extern" and "static" is
 * extremely complex due to the subtle interactions of those keywords, and the
 * many compilers failing to follow the ISO standard properly.
 * 
 * - In ISO C89, "inline" did not exist.
 * - In ISO C99 and above, an "inline" declaration requires the presence of a
 *   definition in the same translation unit.
 * - MSVC never supported C99, but did support an "__inline" keyword.
 * - GCC 2.95 and above do support the "inline" keyword in gnu89 mode (C89+
 *   extensions) as well as c99 mode and above, and the "__inline__" keyword in
 *   any mode.
 * - GCC < 4.2.0 had different semantics for "inline" and "extern inline" than
 *   the C99 standard required.
 * - GCC >= 4.2.0 support proper "inline" and "extern inline" semantics when
 *   -fgnu89-inline is not set.
 * - Clang emulates GCC.
 */


#ifndef BENZINA_INLINE
# if   __GNUC_STDC_INLINE__                /* GCC >= 4.2.0, C99 semantics, any language version */
#  define BENZINA_INLINE __inline__
# elif __GNUC_GNU_INLINE__                 /* GCC >= 4.2.0, GNU semantics, any language version */
#  define BENZINA_INLINE extern __inline__
# elif __GNUC__                            /* GCC  < 4.2.0, GNU semantics, any language version */
#  define BENZINA_INLINE extern __inline__
# elif __STDC_VERSION__ >= 199901L         /* C99 mode. */
#  define BENZINA_INLINE inline
# elif defined(__cplusplus)                /* C++ mode. */
#  define BENZINA_INLINE inline
# elif defined(__MSVC__)                   /* MSVC */
#  define BENZINA_INLINE __inline
# else                                     /* Unknown. */
#  define BENZINA_INLINE static
# endif
#endif


/* End Include Guards */
#endif
