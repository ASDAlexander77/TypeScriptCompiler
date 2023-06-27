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
#define ANY_AS_DEFAULT 1

#ifdef WIN32
#define WIN_EXCEPTION 1
#else
#define LINUX_EXCEPTION 1
#endif

#define USE_NEW_AS_METHOD true
#define ADD_STATIC_MEMBERS_TO_VTABLE true

//#define ALLOC_ALL_VARS_IN_HEAP 1
#define ALLOC_CAPTURED_VARS_IN_HEAP 1
#define ALLOC_CAPTURE_IN_HEAP 1

//#define DISABLE_CUSTOM_CLASSSTORAGESTORAGE 1

#define ENABLE_RTTI true
#define ALL_METHODS_VIRTUAL true
#define USE_BOUND_FUNCTION_FOR_OBJECTS true
#define MODULE_AS_NAMESPACE true

#define ENABLE_TYPED_GC true

//#define ENABLE_DEBUGINFO_PATCH_INFO true

#define ENABLE_JS_BUILTIN_TYPES true
#define NO_DEFAULT_LIB true

#endif // CONFIG_H_