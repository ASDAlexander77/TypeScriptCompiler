#ifndef CONFIG_H_
#define CONFIG_H_

#if _MSC_VER
#pragma warning(disable : 4062)
#pragma warning(disable : 4834)
#pragma warning(disable : 4996)
#endif

//#define GC_ENABLE 1
//#define TSGC_ENABLE 1

#define ENABLE_ASYNC 1
#define ENABLE_EXCEPTIONS 1

#define USE_SPRINTF 1
#ifndef WIN32
#ifndef USE_SPRINTF
#define USE_SPRINTF 1
#endif
#endif

#define NUMBER_F64 1

#ifdef WIN32
#define WIN_EXCEPTION 1
#else
#define LINUX_EXCEPTION 1
#endif

#define REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION true

//#define ALLOC_ALL_VARS_IN_HEAP 1
#define ALLOC_CAPTURED_VARS_IN_HEAP 1
#define ALLOC_CAPTURE_IN_HEAP 1
// TODO: if I uncomment it, it will create errors in capture vars. calls. find out why? (wrong size of buffers?)
//#define ALLOC_TRAMPOLINE_IN_HEAP 1

#endif // CONFIG_H_