// -mm=own rejects: `k` would be a second owner of the block `a` owns, and `a = new C(2)` frees
// what `k` still reads. A global cannot borrow.
class C {
    constructor(public x: number) {}
}

let a = new C(1);
let k = a;

function main() {
    a = new C(2);
    print(k.x);
}
