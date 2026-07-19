import './export_object_literal_structural_typed_params'

// Casts the imported structurally-typed VALUE to a single-method interface in
// the importer, exercising a PARAMETERIZED (not zero-arg) method - extends
// export/import_object_literal_structural_typed.ts's coverage (that test's
// inc() took no arguments).
var acc: A.Accumulator = <A.Accumulator>A.acc;

acc.add(3);
print(acc.total);
assert(acc.total == 3);

acc.add(4);
print(acc.total);
assert(acc.total == 7);

print("done.");
