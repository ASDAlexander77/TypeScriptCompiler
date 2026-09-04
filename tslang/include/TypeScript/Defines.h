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
// Marks a local's storage as holding a reference the scope owns, so that assigning through it
// hands the count over rather than dropping a reference nobody took. Only variable
// declarations set it, which is what keeps parameters and fields - references the frame
// borrows rather than owns - out of the assignment path. See MLIRGen's takeOwnershipOfLocal.
#define OWNED_LOCAL_ATTR_NAME "__owned"

// Marks an operation whose result already carries a reference the receiver is expected to take
// over, rather than one it must retain for itself. `new C()` is the case that matters: it lowers
// to a call of the generated `C..new`, and every function retains its result before returning
// (§9.24), so the value arrives owned. A receiver that retained it again would be one owner
// above the truth, which is exactly the leak §9.25 removes.
//
// Only set where the producer is known to retain - never inferred from a call being a call. A
// runtime or builtin helper, or a function imported from a module built before that convention,
// returns a heap value without any retain, and treating one of those as owned would skip a
// retain nobody performed and free live memory.
#define OWNED_RESULT_ATTR_NAME "__owned_result"

// Marks an owned local that took its reference by consuming an OWNED_RESULT_ATTR_NAME value
// instead of by retaining. The slot still releases at every scope exit - that release is what
// gives the consumed reference back - so the pair is still balanced, but there is no
// `ts.RetainSlot` to pair the release with. The ownership verifier reads this attribute as the
// retain it stands in for.
#define OWNED_LOCAL_CONSUMED_ATTR_NAME "__owned_consumed"

// Marks an OWNED_RESULT_ATTR_NAME operation whose reference some receiver has taken over, so the
// +1 it produced is now somebody's to give back. Set at each of the receiving sites (§9.25) and
// by OwnedReturnConsumptionPass (§9.27) when it removes a receiver's retain.
//
// Its absence is what identifies a discarded temporary: an operation that produced a reference
// nothing took. `f();` on its own, and - far more commonly - a call result used as an argument
// and then dropped, which is what expression-shaped code is made of. See §9.30.
#define OWNED_RESULT_CONSUMED_ATTR_NAME "__owned_result_consumed"
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
// A shared library records the memory model it was built under as an exported data symbol
// named "__tsmm_<model>_<file>_<hash>" - the model is in the NAME, so an importer reads it by
// enumerating symbols and never has to load the data. Deliberately not "__decls"-prefixed, so
// it can never reach the declaration re-parser.
//
// Objects allocated by a module built under a different model must not be freed by this one:
// see docs/reference-counting-evaluation.md section 4. A missing marker means a module built
// before this existed, which is always garbage-collected.
#define SHARED_LIB_MEMORY_MODEL "__tsmm_"
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

// An interface value is { vtable, this, type } - the first two share the indexes above with
// every other pair-shaped value. The third is the runtime type tag of whatever `this` points
// at, which an interface needs for the same reason an `any` box does: the interface type
// carries only a name, so the layout behind `this` is not recoverable from it. With the tag
// there, an interface can be released and retained like anything else, through the concrete
// type's own routines (see OwnershipRoutineLogic and section 9.31).
//
// Null when `this` owns no heap memory - a null interface, or one made from a value that
// carries nothing.
#define INTERFACE_TYPE_INDEX 2

#define ARRAY_DATA_INDEX 0
#define ARRAY_SIZE_INDEX 1

#define OPTIONAL_VALUE_INDEX 0
#define OPTIONAL_HASVALUE_INDEX 1

#define UNION_TAG_INDEX 0
#define UNION_VALUE_INDEX 1

#define ANY_SIZE 0
#define ANY_TYPE 1
#define ANY_DATA 2

// Every heap block reserves one index-sized word in front of its payload (see
// LLVMCodeHelperBase::getHeapBlockHeaderSize). Static blocks - string literals - carry the
// same word, set to this marker, so that a payload pointer is one shape whether it names
// heap or static storage and a release can tell the two apart before calling free.
//
// All bits set, so the marker is the same value whatever the word size or endianness, and it
// is not a value a real count reaches. It is deliberately not zero: a zeroed word is what a
// fresh heap block reads.
//
// Under `-mm=rc` the word holds a live reference count: _MemoryAlloc writes zero into it, and
// a block is freed when a release takes it back to zero (§9.6, §9.24). Under `gc` and `none`
// nothing reads it and nothing writes it - only the static side below is pinned in every model,
// because it is the side that changes a global's layout and so cannot be retrofitted without an
// ABI break. See docs/reference-counting-evaluation.md sections 9.5 and 9.24.
#define HEAP_BLOCK_IMMORTAL -1

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
// Fields may be added, but never reordered or resized, and a new one goes immediately before
// the block header, which has to stay last - see TYPE_DESCR_BLOCK_HEADER.
#define TYPE_DESCR_KIND 0
#define TYPE_DESCR_RESERVED 1
// Address of the type's release routine, or null when the type owns no heap memory - null
// says "nothing to release", not "unknown". Generated by OwnershipRoutineLogic.
#define TYPE_DESCR_RELEASE 2
// Address of the type's retain routine, on the same terms as the release slot. Both exist
// because a tagged union carries its payload inline: copying or dropping one has to retain
// or release a value whose type is only known at run time, and the tag is what knows it.
#define TYPE_DESCR_RETAIN 3
// The block header, last so that it sits immediately in front of the name bytes. A tag is a
// `typeof` result, and `typeof x` can be assigned to a `string` and released like any other
// string - so a tag has to look like a payload with an immortal block header, exactly as a
// string literal does, on top of being a name preceded by a descriptor. Both reads work off
// the same pointer: `tag - sizeof(header)` is the marker, `tag - sizeof(record)` the record.
#define TYPE_DESCR_BLOCK_HEADER 4

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