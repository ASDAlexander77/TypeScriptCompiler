import './export_object_literal_structural_typed'

// Casting the imported structurally-typed VALUE to a method-bearing interface
// in the importing module. Regression coverage for two bugs:
// - the hybrid_func -> func cast crash in CastLogicHelper (PR #256);
// - the interface-cast clone reordering fields to interface (methods-first)
//   order, which silently misread `count` through the vtable's field offset
//   and made inc() corrupt the funcptr slot instead of incrementing.
// Note: this cast clones the object (see the compiler warning), so mutations
// through `c` are visible via `c` but not via A.counterObj.
var c: A.Counter = <A.Counter>A.counterObj;
print(c.count);
assert(c.count == 0);
c.inc();
print(c.count);
assert(c.count == 1);
c.inc();
print(c.count);
assert(c.count == 2);
print("done.");
