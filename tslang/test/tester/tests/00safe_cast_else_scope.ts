// Narrowing a field in an else branch (or a conditional's false side) must end with that branch.
// It leaked into the rest of the function and into the next method compiled, which then read a
// value of a discarded region: a crash, or a bogus "saving to constant object" (#231).

class N {
    v: number = 1;
}

class L {
    head: N | null = null;

    ifElse(): number {
        if (this.head === null) {
            return 0;
        } else {
            return this.head.v;
        }
    }

    ternary(): number {
        return this.head === null ? 0 : this.head.v;
    }

    afterIfElse(): boolean {
        let r = 0;
        if (this.head === null) {
            r = 1;
        } else {
            r = 2;
        }

        return this.head === null;
    }

    isEmpty(): boolean {
        return this.head === null;
    }
}

function main() {
    const l = new L();
    assert(l.ifElse() == 0);
    assert(l.ternary() == 0);
    assert(l.afterIfElse());
    assert(l.isEmpty());

    l.head = new N();
    assert(l.ifElse() == 1);
    assert(l.ternary() == 1);
    assert(!l.afterIfElse());
    assert(!l.isEmpty());

    print("done.");
}
