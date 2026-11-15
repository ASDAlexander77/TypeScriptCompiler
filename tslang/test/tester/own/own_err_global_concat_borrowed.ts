// -mm=own rejects: `id` returns a borrow of its argument, the sum `"a" + "b"` the initializer
// frees once the call is made, so `B` would hold a freed string (TSLANG 85: a concatenation given
// to a call in a global's initializer is moved into a callee that keeps it, or freed after it).
function id(s: string): string {
    return s;
}

const B = id("a" + "b") + "c";

function main() {
    print(B);
}
