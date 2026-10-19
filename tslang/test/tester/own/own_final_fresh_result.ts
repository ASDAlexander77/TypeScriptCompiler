// -mm=own: a method of a @final class is called directly, so its result is fresh, as a direct
// call's is: it may be compared with null and returned, or widened into a union and returned. A
// virtual call's result is not (another module could override the method), which is what kept
// the default library's RegExp.exec and Date[Symbol.toPrimitive] from compiling under own.
export class Res {
    constructor(public v: int) {}
}

@final
export class Re {
    exec(s: string): Res | null {
        if (s.length > 3) {
            return new Res(s.length);
        }

        return null;
    }
}

@final
export class D {
    t = 1;

    toString() {
        return "x" + this.t;
    }

    prim(h: string): string | number {
        if (h === "number") {
            return <number>this.t;
        }

        return this.toString();
    }
}

export function find(re: Re, s: string): int {
    const r = re.exec(s);
    if (r == null) {
        return 0;
    }

    return r.v;
}

function churn() {
    const junk: Res[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Res(-1));
    }
}

function main() {
    const re = new Re();
    const d = new D();
    let t = 0;
    for (let i = 0; i < 10000; i++) {
        t += find(re, i % 3 == 0 ? "abcd" : "ab");
        const p = d.prim("string");
        churn();
        if (typeof p == "string") {
            t += p.length;
        }
    }

    assert(t == 3334 * 4 + 10000 * 2);
    print("done.");
}
