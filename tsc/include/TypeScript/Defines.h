#ifndef DEFINES_H_
#define DEFINES_H_

#define IDENTIFIER_ATTR_NAME "identifier"
#define VIRTUALFUNC_ATTR_NAME "__virt"
#define GENERIC_ATTR_NAME "__generic"
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
#define CALL_FIELD_NAME ".call"
#define THIS_ALIAS ".this"
#define GENERATOR_STEP ".step"
#define GENERATOR_SWITCHSTATE ".switchstate"
#define GENERATOR_STATELABELPREFIX ".state"

#define TO_STRING "toString"
#define SYMBOL_TO_STRING_TAG "toStringTag"
#define SYMBOL_ITERATOR "iterator"
#define SYMBOL_ASYNC_ITERATOR "asyncIterator"
#define ITERATOR_NEXT "next"
#define SYMBOL_HAS_INSTANCE "hasInstance"
#define SYMBOL_TO_PRIMITIVE "toPrimitive"
#define SYMBOL_DISPOSE "dispose"

#define SHARED_LIB_DECLARATIONS "__decls"
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

#define DEFAULT_LIB_DIR "defaultlib"
#define DEFAULT_LIB_NAME "TypeScriptDefaultLib"

#define DEBUG_SCOPE "current"
#define CU_DEBUG_SCOPE "compileUnit"
#define FILE_DEBUG_SCOPE "file"
#define SUBPROGRAM_DEBUG_SCOPE "function"
#define BLOCK_DEBUG_SCOPE "block"
#define NAMESPACE_DEBUG_SCOPE "block"

#endif // DEFINES_H_