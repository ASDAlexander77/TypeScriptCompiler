// A global declared in top-level code owns what it is given: `const t = new C()` there stored the
// instance with no reference of its own, and the reference was then given back at the end of the
// block that made it, leaving the global pointing at a freed block. A `var` in a nested block is
// still a global, and the rest of the top-level code reads it after that block has ended.
class Holder {
    v: number;
    s: string;

    constructor(v: number) {
        this.v = v;
        this.s = "holder-" + v;
    }
}

function churn() {
    let keep: Holder[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Holder(i));
    return keep.length;
}

function readAll() {
    return held.v + later.v + text.length;
}

const held = new Holder(7);
let later = new Holder(8);
var text = "x" + held.v;

{
    var inBlock = new Holder(9);
}

assert(churn() == 1000);
assert(held.v == 7 && held.s == "holder-7");
assert(later.v == 8 && later.s == "holder-8");
assert(text == "x7");
assert(inBlock.v == 9 && inBlock.s == "holder-9");
assert(readAll() == 17);

// overwritten: the old value is given up
later = new Holder(10);
assert(churn() == 1000);
assert(later.s == "holder-10");

print("done.");
