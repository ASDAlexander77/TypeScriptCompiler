// A global initialized with another global's value (`let k = a`), or with a field of one
// (`let kf = h.c`), keeps that block alive when the global it was read from is given a new value.
// Under rc the read took no count of its own: the overwrite freed what the copy still read. Each
// read follows allocations of the same size, so a freed block is reused.
class C {
    tag = "";
    constructor(public x: number) {
        this.tag = "c" + x;
    }
}

class H {
    c = new C(3);
}

class Junk {
    a = -1;
    b = -2;
    tag = "junk";
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

let a = new C(1);
let k = a;
const h = new H();
let kf = h.c;
let s = "s" + a.x;
let ks = s;

function main() {
    a = new C(10);
    h.c = new C(30);
    s = "t" + a.x;
    churn();
    assert(k.x == 1 && k.tag == "c1", "a copy of a global");
    assert(kf.x == 3 && kf.tag == "c3", "a copy of a global's field");
    assert(ks == "s1", "a copy of a global string");
    assert(a.x == 10 && h.c.x == 30 && s == "t10", "the globals' new values");
    print("done.");
}
