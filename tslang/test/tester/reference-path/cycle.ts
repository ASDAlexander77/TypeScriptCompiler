// cycle_a and cycle_b reference each other. Loading them used to never end.
/// <reference path="cycle_a.d.ts" />

const a: CycleA = [1];
const b: CycleB = [2];
assert(a[0] + b[0] == 3);

print("done.");
