// TSLANG_ISSUES 47: the test reads another field than the store writes: not a fill, and the second
// call frees what the first one's result borrows.
class Q {
    constructor(public h: number) {}
}

class Dev {
    cached: Q | undefined;
    other: Q | undefined;
    getQ(): Q {
        if (!this.other) this.cached = new Q(1);
        return this.cached!;
    }
}

function main() {
    const d = new Dev();
    const a = d.getQ();
    const b = d.getQ();
    print(a.h, b.h);
}
