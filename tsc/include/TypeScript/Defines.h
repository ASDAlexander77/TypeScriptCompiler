#ifndef DEFINES_H_
#define DEFINES_H_

#define IDENTIFIER_ATTR_NAME "identifier"
#define VIRTUALFUNC_ATTR_NAME "__virt"
#define GENERIC_ATTR_NAME "__generic"
#define RETURN_VARIABLE_NAME ".return"
#define CAPTURED_NAME ".captured"
#define LCAPTURED_NAME L".captured"
#define LABEL_ATTR_NAME "label"
#define UNDEFINED_NAME "undefined"
#define INFINITY_NAME "Infinity"
#define NAN_NAME "NaN"
#define THIS_NAME "this"
#define LTHIS_NAME L"this"
#define SUPER_NAME "super"
#define STATIC_CONSTRUCTOR_NAME "static_constructor"
#define CONSTRUCTOR_NAME "constructor"
#define CONSTRUCTOR_TEMPVAR_NAME ".ctor"
#define VTABLE_NAME ".vtbl"
#define LVTABLE_NAME L".vtbl"
#define LCONSTRUCTOR_NAME L"constructor"
#define LCONSTRUCTOR_TEMPVAR_NAME L".ctor"
#define RTTI_NAME ".rtti"
#define LRTTI_NAME L".rtti"
#define INSTANCEOF_NAME ".instanceOf"
#define LINSTANCEOF_NAME L".instanceOf"
#define INSTANCEOF_PARAM_NAME "rttiParam"
#define LINSTANCEOF_PARAM_NAME L"rttiParam"
#define MAIN_ENTRY_NAME "main"
#define TS_NEST_ATTRIBUTE "ts.nest"
#define THIS_TEMPVAR_NAME ".this"
#define LTHIS_TEMPVAR_NAME L".this"
#define EXPR_TEMPVAR_NAME ".expr"
#define LEXPR_TEMPVAR_NAME L".expr"
#define TS_GC_ATTRIBUTE "ts.gc"
#define TYPESCRIPT_GC_NAME "tsgc"
#define GLOBAL_CONSTUCTIONS_NAME "llvm.global_ctors"

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

#endif // DEFINES_H_