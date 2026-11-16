// A local moved on some ways through a loop's body and released on the others, where another way
// leaves the loop by `break`: the break releases it once - its own release - and the release on the
// ways that do not move it is not placed on the edge out of the loop as well.
class N {
    constructor(public x: number, public name: string) {}
}

class H {
    n: N | null = null;
}

@noinline
function mk(i: number): N {
    return new N(i, "n" + i);
}

function keep(h: H, p: N) {
    h.n = p;
}

function run(stopAt: number): number {
    const h = new H();
    let seen: number = 0;
    for (let i = 0; i < 10; i++) {
        const n = mk(i);
        if (i == stopAt) {
            break;
        }

        seen += n.x;
        if (i % 2 == 1) {
            keep(h, n);
        }
    }

    return seen + (h.n ? h.n.x * 100 : 0);
}

function main() {
    for (let round = 0; round < 200; round++) {
        assert(run(4) == 0 + 1 + 2 + 3 + 300, "stops at 4");
        assert(run(7) == 21 + 500, "stops at 7");
        assert(run(100) == 45 + 900, "no stop");
    }

    print("done.");
}
