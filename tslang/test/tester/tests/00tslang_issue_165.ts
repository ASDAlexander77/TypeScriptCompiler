// TSLANG_ISSUES 165: a subclass object given to a union of its bases is the nearest base the union has (it was
// "none of the union's types"), and a private name written with an escape (`#\u{78}`) is the name it spells.
class Sq {
    constructor(public size: number) {}
}

class Ci {
    constructor(public r: number) {}
}

class Big extends Ci {
    constructor() {
        super(9);
    }
}

class Huge extends Big {}

function make(n: number): Sq | Ci {
    if (n > 1) return new Huge();
    if (n > 0) return new Big();
    return new Sq(2);
}

function area(s: Sq | Ci) {
    if (s instanceof Ci) return s.r * s.r;
    return s.size * s.size;
}

class P {
    #\u{78} = 4;
    #y1 = 5;

    get() {
        return this.#x + this.#y1;
    }
}

function main() {
    const b = make(1);
    assert(b instanceof Big && b instanceof Ci && !(b instanceof Sq), "Big is a Ci");
    assert(area(b) == 81, "Big's r");

    const h = make(2);
    assert(h instanceof Huge && h instanceof Ci, "Huge is a Ci");

    const s = make(0);
    assert(s instanceof Sq && area(s) == 4, "Sq");

    let total: number = 0;
    for (let i = 0; i < 1000; i++) total += area(make(i % 3));
    assert(total == 334 * 4 + 666 * 81, "in a loop");

    assert(new P().get() == 9, "escaped private names");
    print("done.");
}
