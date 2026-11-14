// TSLANG_ISSUES 47: `fill` only fills `cached`, so its own calls free nothing - but called between
// `bad`'s test and its store, it may have filled the field, and the store then frees what `fill`
// returned. A call in between is judged by facts that count every fill as a drop.
class Q {
    constructor(public h: number) {}
}

class Dev {
    cached: Q | undefined;
    fill(): Q {
        if (!this.cached) this.cached = new Q(1);
        return this.cached;
    }

    bad(): number {
        if (!this.cached) {
            const x = this.fill();
            this.cached = new Q(2);
            return x.h;
        }

        return 0;
    }
}

function main() {
    const d = new Dev();
    print(d.bad());
}
