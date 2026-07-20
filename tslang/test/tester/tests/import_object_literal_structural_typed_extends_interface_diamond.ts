import './export_object_literal_structural_typed_extends_interface_diamond'

let combined: M2.Combined = <M2.Combined>M2.rawCombined;

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
