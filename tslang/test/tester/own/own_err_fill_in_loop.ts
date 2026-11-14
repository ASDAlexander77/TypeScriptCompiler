// TSLANG_ISSUES 47: the field tested empty once, outside a loop, and stored into on each round: the
// second round frees what the first stored. Not a fill.
class Q {
    constructor(public h: number) {}
}

class Dev {
    cached: Q | undefined;
    getQ(): Q {
        const r = this.cached;
        for (let i = 0; i < 2; i++) {
            if (!r) this.cached = new Q(i);
        }

        return this.cached!;
    }
}

function main() {
    const d = new Dev();
    const a = d.getQ();
    const b = d.getQ();
    print(a.h, b.h);
}
