import './export_class_extends_multilevel'

class C extends M.B {
    c: number = 3;

    addC(n: number): void {
        this.c = this.c + n;
    }

    describe(): string {
        return `C:${this.c}/${super.describe()}`;
    }
}

function main() {
    const c = new C();

    c.addA(10);
    c.addB(20);
    c.addC(30);

    assert(c.a == 11);
    assert(c.b == 22);
    assert(c.c == 33);

    assert(c.describe() == "C:33/B:22/A:11");

    // virtual dispatch through each ancestor-typed reference must still
    // reach the most-derived override
    const asB: M.B = c;
    assert(asB.describe() == "C:33/B:22/A:11");

    const asA: M.A = c;
    assert(asA.describe() == "C:33/B:22/A:11");

    print("done.");
}
