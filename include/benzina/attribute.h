/* Include Guards */
#ifndef INCLUDE_BENZINA_ATTRIBUTE_H
#define INCLUDE_BENZINA_ATTRIBUTE_H



/* Visibility: Import/Export/Hidden */
#if   defined(_WIN32) || defined(__CYGWIN__)
# define BENZINA_ATTRIBUTE_EXPORT __declspec(dllexport)
# define BENZINA_ATTRIBUTE_IMPORT __declspec(dllimport)
# define BENZINA_ATTRIBUTE_HIDDEN
#elif __GNUC__ >= 4
# define BENZINA_ATTRIBUTE_EXPORT __attribute__((visibility("default")))
# define BENZINA_ATTRIBUTE_IMPORT __attribute__((visibility("default")))
# define BENZINA_ATTRIBUTE_HIDDEN __attribute__((visibility("hidden")))
#else
# define BENZINA_ATTRIBUTE_EXPORT
# define BENZINA_ATTRIBUTE_IMPORT
# define BENZINA_ATTRIBUTE_HIDDEN
#endif


/* Used */
#if __GNUC__
# define BENZINA_ATTRIBUTE_USED __attribute__((used))
#else
# define BENZINA_ATTRIBUTE_USED
#endif

/* Unused */
#if __GNUC__
# define BENZINA_ATTRIBUTE_UNUSED __attribute__((unused))
#else
# define BENZINA_ATTRIBUTE_UNUSED
#endif

/* Pure (depends only on arguments and only reads global variables) */
#if __GNUC__
# define BENZINA_ATTRIBUTE_PURE __attribute__((pure))
#else
# define BENZINA_ATTRIBUTE_PURE
#endif

/* Const macro (depends only on arguments and doesn't use globals) */
#if __GNUC__
# define BENZINA_ATTRIBUTE_CONST __attribute__((const))
#else
# define BENZINA_ATTRIBUTE_CONST
#endif

/* Deprecated */
#if __GNUC__
# define BENZINA_ATTRIBUTE_DEPRECATED __attribute__((deprecated))
#else
# define BENZINA_ATTRIBUTE_DEPRECATED
#endif

/* Constructor & Destructor function */
#if __GNUC__
# define BENZINA_ATTRIBUTE_CONSTRUCTOR __attribute__((constructor))
# define BENZINA_ATTRIBUTE_DESTRUCTOR  __attribute__((destructor))
#else
# define BENZINA_ATTRIBUTE_CONSTRUCTOR
# define BENZINA_ATTRIBUTE_DESTRUCTOR
#endif

/* Always-inline function */
#ifdef __MSVC__
# define BENZINA_ATTRIBUTE_ALWAYSINLINE __forceinline
#elif __GNUC__
# define BENZINA_ATTRIBUTE_ALWAYSINLINE __attribute__((always_inline))
#else
# define BENZINA_ATTRIBUTE_ALWAYSINLINE
#endif

/* No-inline function */
#ifdef __MSVC__
# define BENZINA_ATTRIBUTE_NOINLINE __declspec(noinline)
#elif __GNUC__
# define BENZINA_ATTRIBUTE_NOINLINE __attribute__((noinline))
#else
# define BENZINA_ATTRIBUTE_NOINLINE
#endif

/* Section */
#if __GNUC__
# define BENZINA_ATTRIBUTE_SECTION(sectionname) __attribute__((section(sectionname)))
#else
# define BENZINA_ATTRIBUTE_SECTION(sectionname)
#endif

/* Packed */
#if __GNUC__
# define BENZINA_ATTRIBUTE_PACKED __attribute__((packed))
#else
# define BENZINA_ATTRIBUTE_PACKED
#endif

/* Aligned */
#if __GNUC__
# define BENZINA_ATTRIBUTE_ALIGNED(n) __attribute__((aligned(n)))
#else
# define BENZINA_ATTRIBUTE_ALIGNED(n)
#endif


/* End Include Guards */
#endif
