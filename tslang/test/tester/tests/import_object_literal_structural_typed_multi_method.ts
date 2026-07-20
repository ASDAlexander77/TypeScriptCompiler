import './export_object_literal_structural_typed_multi_method'

// Casts the imported structurally-typed VALUE to a MULTI-method interface,
// calling each method to exercise every vtable slot - see
// export_object_literal_structural_typed_multi_method.ts for what this covers.
var acc: A.Accumulator = <A.Accumulator>A.acc;

acc.add(3);
print(acc.total);
assert(acc.total == 3);

acc.addTwice(2);
print(acc.total);
assert(acc.total == 7);

print(acc.scaled(2));
assert(acc.scaled(2) == 14);

print("done.");
