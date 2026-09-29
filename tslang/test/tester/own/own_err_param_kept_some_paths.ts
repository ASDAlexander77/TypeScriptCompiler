// -mm=own, phase 4 rejects: `keepIf` keeps its parameter only when `f` is true, and has nothing
// to release it with on the other path.
class C {
    constructor(public x: number) {}
}

class H {
    c: C | null = null;
}

function keepIf(h: H, c: C, f: boolean) {
    if (f) {
        h.c = c;
    }
}

function main() {
    const h = new H();
    keepIf(h, new C(1), true);
    print(h.c!.x);
}
