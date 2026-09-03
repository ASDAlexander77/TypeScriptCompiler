#ifndef DEFINES_H_
#define DEFINES_H_

#define IDENTIFIER_ATTR_NAME "identifier"
#define BUILTIN_FUNC_ATTR_NAME "__builtin"
#define GENERIC_ATTR_NAME "__generic"
#define ATOMIC_ATTR_NAME "__atomic"
#define ORDERING_ATTR_NAME "__ordering"
#define SYNCSCOPE_ATTR_NAME "__syncscope"
#define VOLATILE_ATTR_NAME "__volatile"
#define NONTEMPORAL_ATTR_NAME "__nontemporal"
#define INVARIANT_ATTR_NAME "__invariant"
#define INSTANCES_COUNT_ATTR_NAME "InstancesCount"
#define RETURN_VARIABLE_NAME ".return"
#define CAPTURED_NAME ".captured"
#define LABEL_ATTR_NAME "label"
#define UNDEFINED_NAME "undefined"
#define INFINITY_NAME "Infinity"
#define NAN_NAME "NaN"
#define THIS_NAME "this"
#define SUPER_NAME "super"
#define STATIC_NAME "static"
#define STATIC_CONSTRUCTOR_NAME "static_constructor"
#define CONSTRUCTOR_NAME "constructor"
#define CONSTRUCTOR_TEMPVAR_NAME ".ctor"
#define VTABLE_NAME ".vtbl"
#define RTTI_NAME ".rtti"
#define SIZE_NAME ".size"
#define INSTANCEOF_NAME ".instanceOf"
#define INSTANCEOF_PARAM_NAME "rttiParam"
#define MAIN_ENTRY_NAME "main"
#define TS_NEST_ATTRIBUTE "ts.nest"
#define THIS_TEMPVAR_NAME ".this"
#define EXPR_TEMPVAR_NAME ".expr"
#define TS_GC_ATTRIBUTE "ts.gc"
#define TYPESCRIPT_GC_NAME "tsgc"
#define GLOBAL_CONSTUCTIONS_NAME "llvm.global_ctors"
#define TYPE_BITMAP_NAME ".type_bitmap"
#define TYPE_DESCR_NAME ".type_descr"
#define NEW_METHOD_NAME ".new"
#define NEW_CTOR_METHOD_NAME ".new_ctor"
#define LENGTH_FIELD_NAME "length"
#define INDEX_ACCESS_FIELD_NAME ".index"
#define INDEX_ACCESS_GET_FIELD_NAME "get"
#define INDEX_ACCESS_SET_FIELD_NAME "set"
#define CALL_FIELD_NAME ".call"
#define THIS_ALIAS ".this"
#define GENERATOR_STEP ".step"
#define GENERATOR_SWITCHSTATE ".switchstate"
#define GENERATOR_STATELABELPREFIX ".state"

#define MLIR_GCTORS "__mlir_gctors"

#define TO_STRING "toString"
#define SYMBOL_TO_STRING_TAG "toStringTag"
#define SYMBOL_ITERATOR "iterator"
#define SYMBOL_ASYNC_ITERATOR "asyncIterator"
#define ITERATOR_NEXT "next"
#define SYMBOL_HAS_INSTANCE "hasInstance"
#define SYMBOL_TO_PRIMITIVE "toPrimitive"
#define SYMBOL_DISPOSE "dispose"

// we are using 3 underscore as this is feature of ts parser to add _ to __variables
#define SHARED_LIB_DECLARATIONS_FILENAME "__decls.ts"
#define SHARED_LIB_DECLARATIONS_2UNDERSCORE "__decls"
#define SHARED_LIB_DECLARATIONS "___decls"
#define DLL_EXPORT "dllexport"
#define DLL_IMPORT "dllimport"

#if __LP64__
#define TRAMPOLINE_SIZE 48
#else
#define TRAMPOLINE_SIZE 40
#endif

#define ATTR(attr) mlir::StringAttr::get(rewriter.getContext(), attr)
#define IDENT(name) mlir::Identifier::get(name, rewriter.getContext())
#define NAMED_ATTR(name, attr) mlir::ArrayAttr::get(rewriter.getContext(), {ATTR(name), ATTR(attr)})

#define DATA_VALUE_INDEX 0
#define THIS_VALUE_INDEX 1

#define ARRAY_DATA_INDEX 0
#define ARRAY_SIZE_INDEX 1

#define OPTIONAL_VALUE_INDEX 0
#define OPTIONAL_HASVALUE_INDEX 1

#define UNION_TAG_INDEX 0
#define UNION_VALUE_INDEX 1

#define ANY_SIZE 0
#define ANY_TYPE 1
#define ANY_DATA 2

// Runtime type descriptor.
//
// The ANY_TYPE slot of an "any" box, and the UNION_TAG_INDEX slot of a tagged union, both
// hold a "type tag": a pointer to the NUL-terminated type name, which is what `typeof`
// returns. That name is stored immediately after a fixed-size descriptor record, so the
// descriptor for a tag is reachable at `tag - sizeof(descriptor)` - the same
// header-in-front-of-payload arrangement used for heap blocks. The trailing name is a byte
// array, so it never needs padding in front of it and that offset is exactly the record
// size on every target.
//
// This makes the layout below a cross-module contract even though each module emits its
// own internal-linkage descriptors: a tag produced by one module is read back by another.
// Fields may be appended, but never reordered or resized.
#define TYPE_DESCR_KIND 0
#define TYPE_DESCR_RESERVED 1
// Reserved for the per-type release routine (docs/reference-counting-evaluation.md step 4).
// Always null today and never called; it exists now so pinning the layout is a decision made
// once rather than a later break of the contract above.
#define TYPE_DESCR_RELEASE 2

// Coarse category of the described type. These correspond one-to-one with the names
// TypeOfOpHelper::typeOfAsString reports, and are derived from that name so the two cannot
// drift apart - see TypeOfOpHelper::typeKindFromName.
#define TYPE_KIND_UNKNOWN 0
#define TYPE_KIND_NUMBER 1
#define TYPE_KIND_STRING 2
#define TYPE_KIND_BOOLEAN 3
#define TYPE_KIND_CHAR 4
#define TYPE_KIND_ARRAY 5
#define TYPE_KIND_TUPLE 6
#define TYPE_KIND_OBJECT 7
#define TYPE_KIND_CLASS 8
#define TYPE_KIND_INTERFACE 9
#define TYPE_KIND_FUNCTION 10
#define TYPE_KIND_SYMBOL 11
#define TYPE_KIND_UNDEFINED 12
#define TYPE_KIND_NULL 13

#define DEFAULT_LIB_DIR "defaultlib"
#define DEFAULT_LIB_NAME "TypeScriptDefaultLib"

// The compiled default-lib binaries (dll/ and lib/) are stored in per-build
// subfolders so debug and release artifacts can ship side by side. The
// declaration files (*.d.ts, generics/) are build-mode independent and stay at
// the defaultlib root.
#define DEFAULT_LIB_BUILD_DIR_RELEASE "release"
#define DEFAULT_LIB_BUILD_DIR_DEBUG "debug"

#define DEBUG_SCOPE "current"
#define CU_DEBUG_SCOPE "compileUnit"
#define FILE_DEBUG_SCOPE "file"
#define SUBPROGRAM_DEBUG_SCOPE "function"
#define BLOCK_DEBUG_SCOPE "block"
#define NAMESPACE_DEBUG_SCOPE "block"
#define DEBUG_SCOPE_FOR_SUBPROGRAM "scope_for_subprogram"

#define FIRST_GLOBAL_CONSTRUCTOR_PRIORITY 100
#define LAST_GLOBAL_CONSTRUCTOR_PRIORITY 1000

#endif // DEFINES_H_