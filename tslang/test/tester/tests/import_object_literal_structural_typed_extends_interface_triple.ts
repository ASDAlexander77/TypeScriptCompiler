import './export_object_literal_structural_typed_extends_interface_triple'

let combined: M3.Combined = <M3.Combined>M3.rawCombined;

combined.addA(10);
assert(combined.a == 11);
print(combined.a);

combined.addB(20);
assert(combined.b == 22);
print(combined.b);

combined.addC(30);
assert(combined.c == 33);
print(combined.c);

combined.addCombined(40);
assert(combined.combined == 44);
print(combined.combined);

print("done.");
