// TSLANG_ISSUES 54: a value merged from a `?:` of fresh values or null is the function's own, so it
// moves - once. A second store of it is a second owner, refused.
class P {
    id: int = 3;
}

function make(n: int): P {
    const p = new P();
    p.id = n;
    return p;
}

class K {
    a: P | null = null;
    b: P | null = null;
    run(c: boolean) {
        const p = c ? make(1) : null;
        this.a = p;
        this.b = p;
    }
}

function main() { new K().run(true); }
