// `this` of a @value constructor read through its fields and methods is read as it is now: a method
// called after a field is set sees it, as does a closure that reads `this.a` and is called here.
@value
class W {
    readonly a: number;
    readonly b: number;
    readonly c: number;
    readonly d: number;

    constructor() {
        this.a = 5;
        this.b = this.twiceA();
        const f = () => this.a + this.b;
        this.c = f();
        this.d = this["a"] + 1;
    }

    twiceA() {
        return this.a * 2;
    }
}

function main() {
    const w = new W();
    print(w.b);
    assert(w.b == 10);
    assert(w.c == 15);
    assert(w.d == 6);
    print("done.");
}
