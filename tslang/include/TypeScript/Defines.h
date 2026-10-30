#ifndef DEFINES_H_
#define DEFINES_H_

#include <string>

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

// Marks a variable's storage (its ts.Variable, or its ts.Global) whose type was inferred from an
// integer literal - `let s = 0` is an s32 - so that storing a fraction into it can be warned about.
// See MLIRGen's warnIfIntegerLiteralVariableTruncates.
#define INTEGER_LITERAL_INFERRED_ATTR_NAME "__int_literal_inferred"

// Marks an operation whose result already carries a reference the receiver is expected to take
// over, rather than one it must retain for itself. `new C()` is the case that matters. Of a class
// from another module it lowers to a call of that class's generated `C..new`, and every function
// retains its result before returning (§9.24), so the value arrives owned. Of a class defined in
// the module it is the `ts.New` itself that carries this mark, and that is what makes the block
// start with one reference under rc (NewOpLowering) - so a pass that put this mark on any other
// `ts.New` would change what its block is born with. A receiver that retained the value again
// would be one owner above the truth, which is exactly the leak §9.25 removes.
//
// Only set where the producer is known to retain - never inferred from a call being a call. A
// runtime or builtin helper, or a function imported from a module built before that convention,
// returns a heap value without any retain, and treating one of those as owned would skip a
// retain nobody performed and free live memory.
#define OWNED_RESULT_ATTR_NAME "__owned_result"

// -mm=own only, set by OwnershipInferencePass for the lowering. On a `ts.Release` of an array: the
// array borrowed its elements (the array a rest parameter is given), so its release frees the
// array and not them.
#define OWN_SHALLOW_RELEASE_ATTR_NAME "__own_shallow_release"

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

// Marks the producer of a `const` that was folded into its value rather than given storage -
// `const a = new C()`. The name and the value are one SSA value, so every later mention of `a`
// is the operation's own result, and a receiver cannot tell whether this mention is the last.
// None that stores it may take the reference over; each retains, and the reference the producer
// carried is given back where §9.30 gives back any other unclaimed one. A `return` or a `delete`
// still takes it over, on every path it appears on: nothing after either reads the name. See
// MLIROwnedReference.h.
#define OWNED_RESULT_NAMED_ATTR_NAME "__owned_result_named"

// On a read through an interface's field that is owned under rc (MLIRGenImpl::InterfaceFieldAccess): a number
// of its own. A `ts.Load` only reads memory, so CSE would fold two reads of the same field into one - one
// reference taken, two given back. The number makes no two of them alike.
#define OWNED_READ_ID_ATTR_NAME "__owned_read"

// -mm=own. On a read through an interface's field that may be filled by a getter (MLIRGenImpl::InterfaceFieldAccess),
// and on the `ts.Release` OwnershipInferencePass gives it when nothing it read escapes its block
// (releaseCellReads). Through a plain slot the read is a borrow, as any field read is (spec 28); through a cell of
// accessors it is the getter's result, a value of its own that nothing else holds. The release frees it on that
// path only (ReleaseOpLowering). A read that escapes is not given one, and leaks on the getter's path.
#define OWN_CELL_READ_ATTR_NAME "__own_cell_read"

// Marks a release MLIRGen put at the end of a hand-written `finally` for the exceptional leg only:
// what the scopes around the `try` owe once the exception carries on past it. The finally region
// is one region in MLIRGen but several copies after TryOpLowering - the copy the landing pad runs
// before resuming, and the copies normal completion, a `return` and a `break` run - and only the
// first leaves the function's scopes behind; on every other copy they are still live, and their
// own exits release them. TryOpLowering keeps these in the unwind copy and erases them from the
// rest.
#define UNWIND_ONLY_ATTR_NAME "__unwind_only"

// The `return` counterpart: the `ts.If` MLIRGen put at the end of a hand-written `finally` around
// what the scopes around the `try` owe - their `using`s to dispose and their locals to release -
// once a `return` from its body or catch clause has run the finally. The `return` itself stops
// short of those scopes, since the finally may still read them (GenContext::finallyGuardScope).
// TryOpLowering keeps it in the copy a `return` runs and erases it from the rest.
#define RETURN_ONLY_ATTR_NAME "__return_only"

// The same for a `break` or `continue` out of the `try` body or catch clause, one `ts.If` per
// place they go. Its value names the jump: JUMP_ONLY_BREAK_KEY or JUMP_ONLY_CONTINUE_KEY followed
// by the label (empty when there is none), as the jump's BreakOp or ContinueOp carries it.
// TryOpLowering keeps it in the copy that jump runs and erases it from the rest.
#define JUMP_ONLY_ATTR_NAME "__jump_only"
#define JUMP_ONLY_BREAK_KEY "break:"
#define JUMP_ONLY_CONTINUE_KEY "continue:"

// Marks a `ts.CreateBoundFunction` whose `this` is a capture box built for it a moment earlier,
// rather than a receiver that belongs to somebody else. Only such a closure owns its `this`, and
// gets the box's type tag at CLOSURE_TYPE_INDEX - a bound method does not own the object it is
// bound to (under rc its receivers count it, see BOUND_OBJECT_TYPE_ATTR_NAME). Set where the
// closure is built and the difference is known.
#define OWNS_CAPTURE_ATTR_NAME "__owns_capture"

// On a `ts.CreateBoundFunction` that binds a method to an object: the object's type, which the
// lowering casts `this` away from. Under rc the bound value counts the object through that type's
// tag at CLOSURE_TYPE_INDEX, as an interface over it does.
#define BOUND_OBJECT_TYPE_ATTR_NAME "__bound_object_type"

// Marks the `ts.Variable` that CaptureOpLowering creates for a capture box. It carries
// `captured = true` only because that is how a variable asks to be allocated in the heap, and
// without this marker it would be indistinguishable from the cell of a captured variable - and
// so would be given a frame's reference it has no owner for. A box's owner is the closure.
#define CAPTURE_BOX_ATTR_NAME "__capture_box"

// On a `ts.CreateArray` or a capture box's `ts.Variable` that stack promotion put on the stack
// (docs/superpowers/specs/2026-10-18-stack-promotion-design.md): its blocks go in the frame, with
// immortal headers, instead of on the heap.
#define STACK_ALLOC_ATTR_NAME "__stack_alloc"

// On a `ts.Try`: the statement has a catch clause. Its catches region may still hold nothing but
// the terminator - an untyped `catch { }` emits no ops - and without this marker it would be
// indistinguishable from the catch-less TryOps that only wrap a scope for its cleanup, so the
// lowering would make no landing pad and the exception would go uncaught.
#define HAS_CATCH_ATTR_NAME "__has_catch"

// -mm=own only, set by OwnershipInferencePass for the lowering. On a `ts.CreateBoundFunction`: the
// closure does not escape, so its box borrows what it captures - the box's routine frees the box
// and nothing it holds. On a `ts.ReleaseCell`: the cell borrows its value (a captured parameter's,
// `this`), so the release frees the cell and leaves the value to its owner.
#define OWN_BORROWS_CAPTURES_ATTR_NAME "__own_borrows_captures"
#define OWN_CELL_BORROWS_ATTR_NAME "__own_cell_borrows"
// -mm=own only, set by OwnershipSignaturePass on the module for the lowering and the inference: the
// fields of a generator's state object that borrow what they hold from the generator's arguments,
// as [[type, [field...]], ...] keyed by the state's tuple type, the `.captured` box's index first.
// Its release routine skips them, and a borrowing `.captured` box frees its cells but not what
// they hold.
#define OWN_BORROWING_FIELDS_ATTR_NAME "ts.own_borrowing_fields"
// [class storage type, field position] of each field a back-reference is stored into (own spec 70)
#define OWN_BACK_REFERENCE_FIELDS_ATTR_NAME "ts.own_back_reference_fields"
// -mm=own only, set by OwnershipSignaturePass on the module, in the same shape: the fields of a
// generator's state object - its locals - read only to be yielded (`const groups = [...r]; yield
// groups`). A yield moves what such a field holds to the caller, so the field never owns it: an
// overwrite gives nothing back, and the state's release routine skips it.
#define OWN_YIELDED_FIELDS_ATTR_NAME "ts.own_yielded_fields"
// -mm=own only, set by OwnershipSignaturePass for the inference and the lowering: on the `ts.New`
// that boxes an object literal, and on every `ts.NewInterface` made of that box, the fields of a
// borrowing record (own spec 32) - those holding a value the record does not own (`this`, a
// parameter, a field read, a local's value). The interface's tag then names a release routine of
// its own, which frees the box and the fields it owns and leaves these to their owners: the routine
// travels with the value, so records made at different sites may meet in one local.
#define OWN_RECORD_BORROWED_FIELDS_ATTR_NAME "__own_record_borrowed_fields"
#define RETURN_VARIABLE_NAME ".return"
#define CAPTURED_NAME ".captured"
#define LABEL_ATTR_NAME "label"
// Windows: a landing pad's catch-pad id, and the id of the catch it is nested in - see tagPad in
// TryOpLowering and ts.internal.eh_pad in Win32ExceptionPass
#define EH_PAD_ID_ATTR_NAME "__eh_pad_id"
#define EH_PAD_PARENT_ATTR_NAME "__eh_pad_parent"
#define UNDEFINED_NAME "undefined"
#define NULL_NAME "null"
#define INFINITY_NAME "Infinity"
#define NAN_NAME "NaN"
#define THIS_NAME "this"
#define SUPER_NAME "super"
#define STATIC_NAME "static"
#define STATIC_CONSTRUCTOR_NAME "static_constructor"
#define CONSTRUCTOR_NAME "constructor"
#define CONSTRUCTOR_TEMPVAR_NAME ".ctor"
#define VTABLE_NAME ".vtbl"
// the vtable an object cast to an interface keeps in its own block (castTupleWithOwnVTableToInterface)
#define OWN_INTERFACE_VTABLE_FIELD_NAME ".ivtbl"
#define RTTI_NAME ".rtti"
#define SIZE_NAME ".size"
#define INSTANCEOF_NAME ".instanceOf"
#define INSTANCEOF_PARAM_NAME "rttiParam"
// Every interface vtable starts with a header ahead of the interface's own members:
// - the runtime type tag of what implements it (INTERFACE_VTABLE_TYPE_SLOT): the type descriptor of
//   the class or the object literal the vtable is for, through which an interface value is released
//   and retained (see OwnershipRoutineLogic and section 9.31). Null when that type owns no heap
//   memory, and in an interface's null vtable (mlirGenNullInterfaceVTable).
// - the `.instanceOf` of what implements it: a class's own `.instanceOf`, or, for an object literal,
//   which is no instance of any class, INSTANCEOF_NONE_NAME, which answers false. That lets
//   `i instanceof C` and `<C>anyHoldingAnInterface` ask an interface value what it is.
// An interface member's slot (InterfaceSymbolRefOp's index) is its index among the members, so the
// vtable slot is that plus INTERFACE_VTABLE_HEADER_SLOTS.
// a class's own vtable starts with its `.instanceOf`
#define CLASS_VTABLE_INSTANCEOF_SLOT 0
#define INTERFACE_VTABLE_TYPE_SLOT 0
#define INTERFACE_VTABLE_INSTANCEOF_SLOT 1
#define INTERFACE_VTABLE_HEADER_SLOTS 2
// the name of the type tag's slot in an interface vtable's tuple
#define INTERFACE_VTABLE_TYPE_NAME ".type"
// on the global of an interface's null vtable (mlirGenNullInterfaceVTable): a null interface is seen
// through it, and nothing implements the interface by it
#define NULL_INTERFACE_VTABLE_ATTR_NAME "__null_interface_vtable"
// on the C++ runtime's type_info vtable a `throw`'s exception tables name (`??_7type_info@@6B@`,
// `_ZTVN10__cxxabiv1...`): a foreign global, but no program value is ever read out of it or stored into it
#define EH_TYPE_INFO_ATTR_NAME "__eh_type_info"
// on a ts.Cast from one interface to another whose members are not where the source has them (an
// interface that does not extend the source): for each slot of the target's vtable, the slot of the
// source's that holds it, or -1 for an optional member the source lacks. The cast then gives the
// value a vtable built from its own one, kept once per content (CastOpLowering).
#define INTERFACE_VTABLE_MAP_ATTR_NAME "__vtable_map"
// An interface `v?: T` field's slot is the field's offset from `this`, or -1 when absent. A class
// stores its `v?: T` as optional<T> - {T, flag}, value first - and its slot carries this bit on top
// of the offset: an access then reads the flag (false is undefined), and a write sets it.
#define INTERFACE_OPTIONAL_STORAGE_TAG(sizeBits) (1LL << ((sizeBits) - 2))
// A function-typed interface field (`fn: () => int`) a class fills with a method (`fn(): int`) has
// no storage in the object: its slot is instead the address of a global cell holding the method,
// plus this bit. An access reads the cell (InterfaceSymbolRefOpLowering), then binds `this` as for
// any function-typed field. The bit is free in every other function-typed slot: an offset of a
// pointer-sized field (or of optional<function>, which starts with one) is pointer-aligned, and so
// is the cell; -1 (absent) is tested before it.
// The cell is {method, flag offset, field offset}, the method first: a class's optional field holding a
// function (`fn?: () => int`, called through a thunk) has its flag at `this` + the flag offset, so a read
// of the interface's optional field (`fn?: () => int`) is absent while the instance's flag is clear; -1
// is always present (a method, or a field that is not optional). The field offset is of the class's
// field the thunk calls, when that field is stored as the interface field's function made a hybrid
// (`f: (v: T) => void` in both): a write through the interface (`h.f = g`) stores there. It is -1
// where nothing can be written - a method, or a field of another type - and the write then stops the
// program (INTERFACE_FUNCTION_FIELD_WRITE_ATTR).
// An object field holding a function the interface can't call with the object as its `this` (a
// closure) is filled the same way, with a thunk calling the field; its cell is always present and
// is not written through.
#define INTERFACE_METHOD_CELL_TAG 1
// A field (not optional, not function-typed) a class fills with accessors (`get prop()`, `set prop(v)`) has no
// storage in the object: its slot is instead the address of a global cell {getter, setter}, either null where
// the class has none, plus this bit - the sign bit, which an offset never has (a symbol plus a constant, so it
// is relocated as any address). A load through it calls the getter with `this`, and a store the setter
// (LoadOpLowering, StoreOpLowering). On a 32-bit target this holds for a cell below 2 GB.
#define INTERFACE_ACCESSOR_CELL_TAG(sizeBits) (1ULL << ((sizeBits) - 1))
// on an InterfaceSymbolRefOp of a `v?: T` field that is stored to: true sets the flag of a tagged slot's
// field, false (a store of undefined) clears it
#define INTERFACE_OPTIONAL_WRITE_ATTR "__optional_write"
// on an InterfaceSymbolRefOp of a function-typed field that is stored to (its result is then a ref to
// the field's function as a hybrid): the address of the class field a tagged slot's cell names, or
// null where the slot can't be written (a method's cell, or an object literal's field)
#define INTERFACE_FUNCTION_FIELD_WRITE_ATTR "__function_field_write"
#define INSTANCEOF_NONE_NAME ".instanceOf.none"
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
// the fields of a class value's storage tuple (ClassValueType), beside NEW_METHOD_NAME
#define CLASS_VALUE_VTABLE_NAME ".vtable"
#define CLASS_VALUE_CONSTRUCTOR_NAME ".constructor"
#define CLASS_VALUE_RTTI_NAME ".rtti"
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
// A library built under -mm=own also exports "__tsown_<file>_<hash>": the names of its exported
// functions that destroy nothing a caller can reach (`__own_no_drops`), one per line. The importer
// reads it beside the marker and keeps the names on its module (SHARED_LIB_OWN_NO_DROPS_ATTR_NAME),
// for the ownership signature pass. Only this fact crosses: a missing one just makes an importer
// under own report more, while the facts a callee's body relies on cannot be seen by an importer
// that re-parses source, and would free twice there.
#define SHARED_LIB_OWN_FACTS "__tsown_"
// Which parameters of a module's functions their bodies only borrow, for stack promotion in a
// program linked statically against it: `__tsborrow_<file>_<hash>`, a NUL-terminated char array
// (not a pointer, so it can be read from an object file without relocations), one line per
// function: `<symbol> <index>...`. See docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md.
#define SHARED_LIB_BORROW_FACTS "__tsborrow_"
// [symbol name, text]: set by BorrowFactsExportPass, made into the symbol by the LLVM lowering
#define SHARED_LIB_BORROW_FACTS_ATTR_NAME "ts.borrow_facts"
// a module attribute, so it has to carry the dialect's prefix
#define SHARED_LIB_OWN_NO_DROPS_ATTR_NAME "ts.own_imported_no_drops"
// A module built under -mm=own also publishes its own facts as a char array, readable from its
// object file without relocations (__tsown_ is a pointer to the text, read through the loaded
// library): `__tsownfacts_<file>_<hash>`, one line per exported function - its name alone when
// it destroys nothing a caller can reach (__own_no_drops), or its name followed by a tab-separated
// `<position> <slot type>` per place it may overwrite when that is all it destroys
// (__own_drop_fields). A program compiled against the module's object reads them with
// --own-facts=<object>: see docs/superpowers/specs/2026-10-25-own-facts-from-objects-design.md.
#define SHARED_LIB_OWN_OBJECT_FACTS "__tsownfacts_"
// [symbol name, text]: set by OwnershipSignaturePass, made into the symbol by the LLVM lowering
#define SHARED_LIB_OWN_OBJECT_FACTS_ATTR_NAME "ts.own_object_facts"
// a module attribute: { name -> [[position, slot type]...] } of the functions --own-facts says
// overwrite only those places
#define SHARED_LIB_OWN_DROP_FIELDS_ATTR_NAME "ts.own_imported_drop_fields"
// ...and [name...] of the functions it says return a value their caller owns, borrowing no argument:
// a line's last field is OWN_OBJECT_FACTS_FRESH_RESULT
#define SHARED_LIB_OWN_FRESH_RESULTS_ATTR_NAME "ts.own_imported_fresh_results"
#define OWN_OBJECT_FACTS_FRESH_RESULT "fresh"
// `@nodrops declare function strncmp(...)`: a function defined outside TypeScript, a C function,
// that destroys nothing a TypeScript caller can reach. -mm=own calls it the way it calls an
// imported function in SHARED_LIB_OWN_NO_DROPS_ATTR_NAME, so a borrow of a field can be given to
// it. A declaration has no body to check, so this is the declaration's word, and only a
// declaration can give it: never on one that may run TypeScript code (it takes a callback) or
// one re-parsed from a TypeScript library, whose functions may destroy what they are given.
#define NO_DROPS_DECORATOR "nodrops"
// The memory model a module is compiled for, a string literal: `if (__MEMORY_MODEL__ === "gc")`
// generates the branch taken only (MLIRGenImpl::foldLiteralCompare)
#define MEMORY_MODEL_IDENTIFIER "__MEMORY_MODEL__"
// ...and the target's OS and architecture, as Node's process.platform and process.arch name them
#define TARGET_OS_IDENTIFIER "__OS__"
#define TARGET_ARCH_IDENTIFIER "__ARCH__"
// The constant op one of them generates: only a compare with it is decided while generating, as
// other values' literal types (a narrowed field's, `~1`'s) need not be what they hold at run time
#define COMPILE_TIME_CONSTANT_ATTR_NAME "ts.compile_time_constant"
#define OWN_DECLARED_NO_DROPS_ATTR_NAME "ts.own_declared_no_drops"
// `@keeps("p")` and `@borrows("this", "p")`: the facts of a function's body that its callers rely
// on (OWN_PARAMS_ATTR_NAME, OWN_RESULT_BOUNDED_ATTR_NAME), said in its signature, so every caller
// knows them - one in another module too, which reads them from the declaration it re-parses.
// `@keeps` names the parameters the function keeps: its callers move them in. `@borrows` names the
// arguments a generator's state object borrows: the caller owns the result, but it may not outlive
// them. `@lends` names the arguments the result is a borrow of (OWN_RESULT_BORROWS_ATTR_NAME): one
// for a plain borrow, whose caller owns nothing, or several for a borrowing record (own spec 32),
// whose caller owns the box; the result may not outlive any of them. -mm=own checks a body against what
// it says; the other memory models ignore all three. Each is a FuncOp attribute of argument indices.
#define KEEPS_DECORATOR "keeps"
#define BORROWS_DECORATOR "borrows"
#define LENDS_DECORATOR "lends"
#define OWN_DECLARED_KEEPS_ATTR_NAME "ts.own_declared_keeps"
#define OWN_DECLARED_BORROWS_ATTR_NAME "ts.own_declared_borrows"
#define OWN_DECLARED_LENDS_ATTR_NAME "ts.own_declared_lends"
// `@record`, beside `@lends`: the result is a borrowing record (own spec 32) - a box the caller owns,
// holding a borrow of each argument `@lends` names - not a borrow of one of them. An importer, which
// sees the declaration without its body, learns it only from this; a function another module can call
// that returns such a record has to say it. A unit attribute on the FuncOp.
#define RECORD_DECORATOR "record"
#define OWN_DECLARED_RECORD_ATTR_NAME "ts.own_declared_record"
// `@copies`: inside this function -mm=own may copy an array whose elements own nothing where it cannot
// prove one owner (own spec 71), as it copies a string anywhere. A copy of an array can be told from
// the array - a change to one is not a change to the other - so only a function that says so makes
// one: reduce's accumulator, which starts as an element. The other models ignore it. A unit attribute.
#define COPIES_DECORATOR "copies"
#define OWN_DECLARED_COPIES_ATTR_NAME "ts.own_declared_copies"
// `@immortalizes` names the parameters the function keeps for good, in an append-only global (a
// keeper, docs/superpowers/specs/2026-10-25-own-immortal-keepers-design.md): its callers make each
// argument immortal. Checked against a body; trusted on a declaration, a C function that keeps a
// callback for good. A FuncOp attribute of argument indices.
#define IMMORTALIZES_DECORATOR "immortalizes"
#define OWN_DECLARED_IMMORTALIZES_ATTR_NAME "ts.own_declared_immortalizes"
// the names of the arguments of a function that says any, for -mm=own's errors about it
#define OWN_DECLARED_PARAM_NAMES_ATTR_NAME "ts.own_declared_param_names"

// The process-heap allocator of a Windows module (TypeScript/ProcessHeap.h), which ProcessHeapPass
// calls instead of the C runtime's. Defined by the async runtime library a program links,
// TypeScriptRuntime.dll and the JIT.
#define PROCESS_HEAP_MALLOC "__tslang_heap_malloc"
#define PROCESS_HEAP_CALLOC "__tslang_heap_calloc"
#define PROCESS_HEAP_REALLOC "__tslang_heap_realloc"
#define PROCESS_HEAP_FREE "__tslang_heap_free"
#define PROCESS_HEAP_ALIGNED_ALLOC "__tslang_heap_aligned_alloc"
#define DLL_EXPORT "dllexport"
#define DLL_IMPORT "dllimport"
#define DLL_NAME "dllname"
#define LINK_NAME "linkname"
// @cppname("int Methods::scaled(int)"): @linkname under the symbol the target's C++ compiler gives
// that declaration (see CppMangler.h)
#define CPP_NAME "cppname"
// A class laid over an existing C++ object: its vtable is the C++ one (see ClassInfo::isCppVTable)
#define CPP_VTABLE "cpp"
// `@final class C`: no class may extend C, so a call on a value of type C is a direct call to the
// method C has (docs/superpowers/specs/2026-10-19-final-classes-design.md)
#define FINAL_DECORATOR "final"
// `@value class C` / `@value interface R`: a type whose values are copied, not shared - no identity,
// no vtable, no base (docs/superpowers/specs/2026-10-24-value-types-design.md)
#define VALUE_DECORATOR "value"
// `@sharedElements class Map<K, V>`: a container that keeps what it is given and hands out
// borrows of it. Under -mm=own each of its type arguments has to be a value, a string (copied) or
// a Shared<T> handle (counted): an object, whose block has one owner, goes through Shared<T>.
// The other memory models ignore it.
#define SHARED_ELEMENTS_DECORATOR "sharedElements"
// On a `ts.ThisVirtualSymbolRef` (a method's) or a `ts.VirtualSymbolRef` (an accessor's): the
// receiver's class is @final, but defined in another module, so the call still goes through the
// vtable. The slot holds the method the reference names, which
// -mm=own calls the way it calls that module's function: its `__own_no_drops` applies.
#define FINAL_RECEIVER_ATTR_NAME "__final_receiver"
// the old name of @cpp, an error now: unknown, it would leave the class an ordinary one
#define CPP_VTABLE_OLD_NAME "cppvtable"
// A @cpp method's C++ return type has a destructor or a copy constructor (see
// mlirGenClassCppVTableHiddenReturn)
#define CPP_NON_TRIVIAL "cppnontrivial"
// @slot(N) on a method of a @cpp class: a C++ virtual method, slot N of the vtable (counted
// from 0, a base class's slots included); a method without it isn't virtual (see MethodInfo::cppSlot)
#define CPP_SLOT "slot"
// The name of a C++ overload after the first one declared, `write#1` (see mlirGenClassCppOverloadNames)
#define CPP_OVERLOAD_SEPARATOR "#"
// On a call (CallIndirect, then CallInternal or Invoke): the index of the argument that is the hidden
// pointer the callee returns a record through, which LowerToLLVM marks `sret` for the target to pass
// it where the C++ ABI does (X8 on AArch64)
#define SRET_ARG_ATTR_NAME "ts.sret_arg"
// On a call: it calls a C++ method (of a @cpp class), whose arguments narrower than 32 bits
// LowerToLLVM marks zeroext / signext, as a C++ caller extends them (clang relies on it)
#define CPP_CALL_ATTR_NAME "ts.cpp_call"
// On a function: a C++ method implemented in tslang (a method with a body of a @cpp class),
// called by C++, whose results narrower than 32 bits LowerToLLVM extends as C++ callers expect
#define CPP_METHOD_ATTR_NAME "ts.cpp_method"

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

// An interface value is { vtable, this }, sharing the indexes above with every other pair-shaped
// value. The runtime type tag of whatever `this` points at, which an interface needs for the same
// reason an `any` box does - the interface type carries only a name, so the layout behind `this`
// is not recoverable from it - is the vtable's first slot (INTERFACE_VTABLE_TYPE_SLOT): the vtable
// is made for that type, the class's or the object literal's.

// A bound or hybrid function value is { func, this, type }, and the third word is there for the
// same reason as an interface's: the `this` of a closure is its capture box, heap-allocated and
// not named anywhere in the function type, so nothing could give it back. The tag makes it
// releasable through the box's own routine.
//
// Null for every function value that is not a closure over captured variables - a plain function
// pointer, a bound method, an interface's method slot - which is what keeps `obj.m` from taking
// ownership of `obj`. Only CreateBoundFunctionOp builds one of these, so there is no path that
// leaves the slot undefined. See section 9.33.
#define CLOSURE_TYPE_INDEX 2

#define ARRAY_DATA_INDEX 0
#define ARRAY_SIZE_INDEX 1
#define ARRAY_CAPACITY_INDEX 2

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
// For a tuple, an id of its layout - its fields' names and types (TypeOfOpHelper::layoutIdOf) -
// for an interface, an id of its name (which interface a union of interfaces holds, every one's tag
// being "interface": ts.IsTypeTagOf), and 0 for anything else. A cast from `any` to an object type compares it with the type's own
// (ts.IsTupleOfLayout): the box says only "tuple", and a tuple of other fields read as the type's
// was garbage (`{ n: 5 }` holds an s32, read as a number). Computed from the type's text, so a
// tuple boxed by one module matches the same type in another.
#define TYPE_DESCR_LAYOUT 1
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

// ...and then per memory model, because a default lib is not model-neutral. Under `-mm=gc` it
// allocates through Boehm and drags libgc in with it; under `-mm=rc` it initialises the block
// header's reference count and follows the +1 return convention (§9.24); under `-mm=none` it
// does neither. Linking one model's library into another model's program is the case §9.7
// warns about, and it is the largest instance of it, since every program that does not pass
// `--no-default-lib` links this one.
//
// The model name is exactly `memoryModelName()`, so the directory and the `-mm=` flag cannot
// drift apart.
#define DEFAULT_LIB_KIND_STATIC "lib"
#define DEFAULT_LIB_KIND_SHARED "dll"

// ...and, outermost under lib/ or dll/, per target, as the LLVM triple names it: arch, vendor, OS
// and environment, each the canonical name LLVM gives the parsed component, so the spellings of
// one target (i686/i386, an Android API level, an MSVC version) share a tree. Every target is
// keyed this way, the host included. There is no fallback to another target's tree: a library
// built for one links into a program for another only to fail on every symbol.
//
// Layout: defaultlib/{lib,dll}/<arch>/<vendor>/<os>/<env>/{debug,release}/{gc,rc,none}/, e.g.
// defaultlib/lib/x86_64/pc/windows/msvc/release/gc or defaultlib/lib/aarch64/unknown/linux/android/
// release/gc. Everything else (*.d.ts, generics/, ...) stays at the defaultlib root, shared by
// every build. Composed in ONE place, getDefaultLibSubDir (tslang/tslang/defaultlib.cpp), which
// the linker, the JIT, --install-default-lib and --print-default-lib-dir all call; the build
// scripts ask --print-default-lib-dir rather than spelling the path out.

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