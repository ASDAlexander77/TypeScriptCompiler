// @strict-null false

let a = [1, 2, 3];

let pa = ReferenceOf(a[0]);

assert(LoadReference(pa[0]) == 1);
assert(LoadReference(pa[1]) == 2);
assert(LoadReference(pa[2]) == 3);

print("done.");