// TSLANG_ISSUES 47 (-mm=own): a lazily cached field could not be returned through a local.
// `let r = this.cached; if (!r) { r = new Q(); this.cached = r; } return r;` returns on both ways
// the object the field holds: a read of the local where it holds what the field holds is read as
// the field. And a store that fills an empty field (`if (!this.cached) this.cached = ...`) frees
// nothing, so a second call of the getter does not end the borrow the first one returned.

class Q {
    constructor(public h: number) {}
}

class Dev {
    cached: Q | undefined;
    other: Q | null = null;
    made = 0;

    // the issue's getter: through a `let`
    viaLet(): Q {
        let result = this.cached;
        if (!result) {
            this.made++;
            result = new Q(1);
            this.cached = result;
        }

        return result;
    }

    // the new one given to the `let` after the field
    viaAssign(): Q {
        let r = this.cached;
        if (!r) {
            this.made++;
            const n = new Q(1);
            this.cached = n;
            r = n;
        }

        return r;
    }

    // check, then return the narrowed `const`; or the new one, once stored
    viaConst(): Q {
        const hit = this.cached;
        if (hit) return hit;
        this.made++;
        const result = new Q(1);
        this.cached = result;
        return result;
    }

    // the field read twice
    viaField(): Q {
        if (!this.cached) {
            this.made++;
            this.cached = new Q(1);
        }

        return this.cached;
    }

    // tested with `=== undefined`, `== null`, and filled in the else branch
    viaEquals(): Q {
        if (this.cached === undefined) {
            this.made++;
            this.cached = new Q(1);
        }

        return this.cached;
    }

    viaNull(): Q {
        if (this.other == null) {
            this.made++;
            this.other = new Q(2);
        }

        return this.other;
    }

    viaElse(): Q {
        if (this.cached) {
            return this.cached;
        } else {
            this.made++;
            this.cached = new Q(1);
            return this.cached;
        }
    }
}

function main() {
    // the borrow of the first result outlives the second call
    const d = new Dev();
    const a = d.viaLet();
    const b = d.viaLet();
    print(a.h, b.h, a === b);
    assert(a === b && d.made === 1);

    const e = new Dev();
    const c1 = e.viaConst();
    const c2 = e.viaConst();
    assert(c1 === c2 && e.made === 1);

    const f = new Dev();
    const f1 = f.viaField();
    const f2 = f.viaField();
    assert(f1 === f2 && f.made === 1);

    const g = new Dev();
    const g1 = g.viaEquals();
    const g2 = g.viaEquals();
    const g3 = g.viaNull();
    const g4 = g.viaNull();
    assert(g1 === g2 && g3 === g4 && g.made === 2);

    const v = new Dev();
    const v1 = v.viaAssign();
    const v2 = v.viaAssign();
    assert(v1 === v2 && v.made === 1);

    const h = new Dev();
    const h1 = h.viaElse();
    const h2 = h.viaElse();
    assert(h1 === h2 && h.made === 1);

    // a getter of one object, then the same of another
    const x = new Dev();
    const y = new Dev();
    const xq = x.viaLet();
    const yq = y.viaLet();
    assert(xq !== yq && xq.h === yq.h);

    print("done.");
}
