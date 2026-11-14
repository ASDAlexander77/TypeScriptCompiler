// TSLANG_ISSUES 64: a value moved into a field of an object already pushed into an array borrows that
// field, so the object popped before the use frees it.
class V {
    n: int;
    constructor(n: int) {
        this.n = n;
    }
}

class F {
    v: V | null = null;
}

class H {
    frames: F[] = [];
}

export function f(h: H, c: boolean): int {
    const fr = new F();
    h.frames.push(fr);
    const v = c ? new V(2) : null;
    fr.v = v;
    h.frames.pop();
    return v ? v.n : 0;
}

function main() { print(f(new H(), true)); }
