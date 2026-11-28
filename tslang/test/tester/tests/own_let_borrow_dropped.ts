// -mm=own: a `let` that borrows a field still holds it where the field is given another object on some path (TSLANG
// 217, the sound shape of 56 compiles): `cur` is `this.a` when `c` is false, read after `this.a` is replaced. An error
// until TSLANG 221: the `let`, assigned again, takes a reference of its own (USER RULING 11-27).
class V {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

class C {
    a: V = new V(1);
    b: V = new V(2);
    run(c: boolean): number {
        let cur = this.a;
        if (c) cur = this.b;
        this.a = new V(5);
        return cur.x;
    }
}

function main() {
    const c = new C();
    if (c.run(false) != 1 || c.run(true) != 2) throw "a borrowed field lost its value";
    print("done.");
}
