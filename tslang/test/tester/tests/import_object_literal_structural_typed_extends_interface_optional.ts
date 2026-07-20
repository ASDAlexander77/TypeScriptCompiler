import './export_object_literal_structural_typed_extends_interface_optional'

let present: M4.Derived = <M4.Derived>M4.rawPresent;
let missing: M4.Derived = <M4.Derived>M4.rawMissing;

assert(present.base == 1);
assert(present.opt == 5);
assert(present.derived == 10);

assert(missing.base == 2);
assert(missing.opt == undefined);
assert(missing.derived == 20);

print(present.opt);
assert(present.opt == 5);

print("done.");
