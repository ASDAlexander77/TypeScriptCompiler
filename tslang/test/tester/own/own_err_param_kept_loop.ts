// -mm=own, phase 4 rejects: `c` is made once, outside the loop, and moved into `keep` on every
// iteration.
class C {
    constructor(public x: number) {}
}

class H {
    c: C | null = null;
}

function keep(h: H, c: C) {
    h.c = c;
}

function main() {
    const h = new H();
    const c = new C(1);
    for (let i = 0; i < 3; i++) {
        keep(h, c);
    }

    print(h.c!.x);
}
