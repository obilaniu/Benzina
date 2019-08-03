/* Include Guard */
#ifndef INCLUDE_BENZINA_CONFIG_H
#define INCLUDE_BENZINA_CONFIG_H


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


/* End Include Guard */
#endif

