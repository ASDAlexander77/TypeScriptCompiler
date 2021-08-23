#ifndef DEFINES_H_
#define DEFINES_H_

#define IDENTIFIER_ATTR_NAME "identifier"
#define RETURN_VARIABLE_NAME ".return"
#define CAPTURED_NAME ".captured"
#define LABEL_ATTR_NAME "label"
#define UNDEFINED_NAME "undefined"
#define THIS_NAME "this"
#define LTHIS_NAME L"this"
#define SUPER_NAME "super"
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

#if __LP64__
#define TRAMPOLINE_SIZE 48
#else
#define TRAMPOLINE_SIZE 40
#endif

#endif // DEFINES_H_