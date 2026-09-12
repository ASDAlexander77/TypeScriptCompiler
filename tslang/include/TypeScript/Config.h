#ifndef CONFIG_H_
#define CONFIG_H_

#if _MSC_VER
#pragma warning(disable : 4062)
#pragma warning(disable : 4834)
#pragma warning(disable : 4996)

// CRT debug
#ifdef _DEBUG
   #define _CRTDBG_MAP_ALLOC
#endif // _DEBUG    

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
// somehow it will error if set to true
#define TUPLE_TYPE_PACKED false
#define UNION_TYPE_PACKED true

#ifdef WIN32
#define WIN_LOADSHAREDLIBS 1
#else
#define LINUX_LOADSHAREDLIBS 1
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

// Typed (precise-heap) allocation of class instances via GC_malloc_explicitly_typed.
//
// Disabled: the per-class pointer bitmap that feeds GC_make_descriptor is generated
// incorrectly, so the "precision" it bought was never real. mlirGenClassTypeBitmap has
// three defects - it shifts right where it means to shift left (so no bit above position
// zero is ever set), it indexes the bitmap array by the field's word index within the
// object instead of that index divided by the word bit count (so it also runs off the end
// of the stack array), and the array is never zeroed because AllocaOpLowering still
// carries its "TODO: call MemSet". The descriptor therefore came from uninitialized stack
// memory, and any pointer field whose bit read as zero was left untraced - a live object
// could be collected. Short tests rarely trigger a collection, which is why this stayed
// latent.
//
// With this off, class instances take the same generic allocation path as everything else
// (NewOp -> NewOpLowering -> MemoryAlloc) and are scanned conservatively, which is what
// they effectively got anyway. That also gives them the heap block header uniformly - see
// docs/reference-counting-evaluation.md sections 4 and 9.1.
//
// Re-enabling requires fixing all three defects in mlirGenClassTypeBitmap first, and then
// shifting every bit by the header size, since descriptor bits are object-base-relative.
#define ENABLE_TYPED_GC false

//#define ENABLE_DEBUGINFO_PATCH_INFO true

#define ENABLE_JS_BUILTIN_TYPES true
//#define ENABLE_JS_TYPEDARRAYS true
//#define ENABLE_JS_TYPEDARRAYS_NOBUILTINS true
#define ENABLE_NATIVE_TYPES true
#define NO_DEFAULT_LIB true

// seems we can't use appending logic at all
//#define SHARED_LIB_DECLARATION_INFO_IS_APPENDABLE true

//#define DBG_INFO_ADD_VALUE_OP true

//#define GENERATE_IMPORT_INFO_USING_D_TS_FILE true

#endif // CONFIG_H_