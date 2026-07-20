import './export_object_literal_untyped_multi_method'

// Direct calls on the imported boxed global, no interface cast - exercises
// every method field, not just the first (see export file's comment).
A.acc.add(3);
print(A.acc.total);
assert(A.acc.total == 3);

A.acc.addTwice(2);
print(A.acc.total);
assert(A.acc.total == 7);

print(A.acc.scaled(2));
assert(A.acc.scaled(2) == 14);

// Also exercise the interface-cast path on the same boxed value.
var acc2: A.Accumulator = <A.Accumulator>A.acc;
print(acc2.total);
assert(acc2.total == 7);
acc2.add(1);
print(acc2.total);
assert(acc2.total == 8);

print("done.");
