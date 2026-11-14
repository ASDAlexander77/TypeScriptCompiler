// TSLANG_ISSUES 47: a store into a field tested empty frees nothing - but not when a call between
// the test and the store may write the field. `clear` fills it, and the store then frees what
// `clear` stored, which the first call's result borrows.
class Q {
    constructor(public h: number) {}
}

class Dev {
    cached: Q | undefined;
    clear() { this.cached = new Q(5); }
    getQ(): Q {
        if (!this.cached) { this.clear(); this.cached = new Q(1); }
        return this.cached;
    }
}

function main() {
    const d = new Dev();
    const a = d.getQ();
    const b = d.getQ();
    print(a.h, b.h);
}
