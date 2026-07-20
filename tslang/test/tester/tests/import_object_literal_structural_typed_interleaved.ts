import './export_object_literal_structural_typed_interleaved'

var acc: A.Accumulator = <A.Accumulator>A.acc;

acc.addBase(5);
print(acc.base);
assert(acc.base == 105);

acc.add(3);
print(acc.total);
assert(acc.total == 3);

acc.addTwice(2);
print(acc.total);
assert(acc.total == 7);

print(acc.scaled(2));
assert(acc.scaled(2) == 14);

print("done.");
