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
 * - In ISO C99 and above, the standard states:
 *       §6.7.4(6) Any function with internal linkage can be an inline function.
 *                 For a function with external linkage, the following
 *                 restrictions apply: If a function is declared with an inline
 *                 function specifier, then it shall also be defined in the same
 *                 translation unit. If all of the file scope declarations for
 *                 a function in a translation unit include the inline function
 *                 specifier without extern, then the definition in that
 *                 translation unit is an inline definition. An inline
 *                 definition does not provide an external definition for the
 *                 function, and does not forbid an external definition in
 *                 another translation unit. An inline definition provides an
 *                 alternative to an external definition, which a translator
 *                 may use to implement any call to the function in the same
 *                 translation unit. It is unspecified whether a call to the
 *                 function uses the inline definition or the external
 *                 definition.
 * - GCC 1.21 and above do support the "inline" keyword in gnu89 mode as a C89
 *   extension as well as C99 mode and above
 * - GCC 2.95 and above support the alternate "__inline__" keyword in any mode.
 * - GCC < 4.3.0 had different semantics for "inline" and "extern inline" than
 *   the C99 standard required, even in C99 mode.
 * - GCC >= 4.3.0 support C99 semantics by default in C99+ modes when
 *   -fgnu89-inline is not set, and GNU semantics in C89 mode.
 * - GCC >= 4.1.3 sets exactly one of __GNUC_STDC_INLINE__ or __GNUC_GNU_INLINE__
 *   to 1 according to whether GNU or C99 semantics are in use.
 *     - GCC 4.2.x does not support C99 semantics, so it will never set
 *       __GNUC_STDC_INLINE__.
 *     - GCC 4.2.x warns about non-static inline functions except if marked with
 *       __attribute__((gnu_inline)) or if -fgnu89-inline is explicitly passed.
 * - Clang emulates GCC 4.2.1, but does support C99 semantics and does set
 *   __GNUC_STDC_INLINE__ under the same conditions as GCC 4.3.0+.
 * - MSVC never supported C99, but did support an "__inline" keyword.
 * - C++ supports the "inline" and "extern inline" keywords with identical
 *   semantics, and duplicate definitions are resolved using the normal
 *   language rules.
 */


#if   __GNUC_STDC_INLINE__                /* GCC >= 4.3.0 or Clang, C99 semantics, any language version */
# define BENZINA_INLINE            __inline__
# define BENZINA_EXTERN_INLINE     extern __inline__
#elif __GNUC_GNU_INLINE__                 /* GCC >= 4.1.3 or Clang, GNU semantics, any language version */
# define BENZINA_INLINE            extern __inline__
# define BENZINA_EXTERN_INLINE     __inline__
#elif defined(__GNUC__) && __GNUC__ <= 4  /* GCC  < 4.1.3, GNU semantics, any language version */
# define BENZINA_INLINE            extern __inline__
# define BENZINA_EXTERN_INLINE     __inline__
#elif __STDC_VERSION__ >= 199901L         /* C99 mode. */
# define BENZINA_INLINE            inline
# define BENZINA_EXTERN_INLINE     extern inline
#elif defined(__cplusplus)                /* C++ mode. */
# define BENZINA_INLINE            inline
# define BENZINA_EXTERN_INLINE     extern inline
#elif defined(__MSVC__)                   /* MSVC. Unlikely to work. */
# define BENZINA_INLINE            __inline
# define BENZINA_EXTERN_INLINE     extern __inline
#else                                     /* Unknown. */
# define BENZINA_INLINE            static
# define BENZINA_EXTERN_INLINE     static
#endif


/* End Include Guards */
#endif
