// @strict-null false

let a = [1, 2, 3];

const pa: Ref<TypeOf<1>> = Ref(a[0]);

assert(Deref(pa[0]) == 1);
assert(Deref(pa[1]) == 2);
assert(Deref(pa[2]) == 3);

print("done.");
