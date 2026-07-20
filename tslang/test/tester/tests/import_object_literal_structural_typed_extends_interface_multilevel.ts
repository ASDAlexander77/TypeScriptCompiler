import './export_object_literal_structural_typed_extends_interface_multilevel'

let obj: M.C = <M.C>M.raw;

obj.addA(10);
assert(obj.a == 11);
print(obj.a);

obj.addB(20);
assert(obj.b == 22);
print(obj.b);

obj.addC(30);
assert(obj.c == 33);
print(obj.c);

let combined: M.Combined = <M.Combined>M.rawCombined;

combined.addLeft(10);
assert(combined.left == 11);
print(combined.left);

combined.addRight(20);
assert(combined.right == 22);
print(combined.right);

combined.addCombined(30);
assert(combined.combined == 33);
print(combined.combined);

print("done.");
