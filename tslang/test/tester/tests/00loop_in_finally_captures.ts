// A loop in a catch clause or a `finally` gives each iteration its own locals. Those clauses put
// their locals in front of the `try` (so both copies of a finally see one slot), and a loop in one
// had them put there too: every iteration shared one slot. A closure capturing it saw only the last
// iteration's value, and under rc and own its cell was given back at the end of the first
// iteration and written by the next - heap corruption (fuzz 945467).

class V {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function mv(i: number): V {
    return new V(i);
}

function f(i: number) {
    if (i > 100) throw i;
}

function inFinally(n: number) {
    const fns: (() => number)[] = [];
    try {
        f(1);
    } finally {
        for (let j = 0; j < n; j++) {
            let v: V = mv(j + 1);
            fns.push((): number => v.x);
        }
    }

    let sum: number = 0;
    for (const c of fns) sum += c();
    return sum;
}

function inCatch(n: number) {
    const fns: (() => number)[] = [];
    try {
        f(1000);
    } catch (e) {
        for (let j = 0; j < n; j++) {
            let v: V = mv(j + 1);
            fns.push((): number => v.x);
        }
    }

    let sum: number = 0;
    for (const c of fns) sum += c();
    return sum;
}

function sharedInFinally(n: number) {
    let sum: number = 0;
    try {
        f(1);
    } finally {
        let j = 0;
        while (j < n) {
            let sh: Shared<V> = new Shared<V>(mv(j + 1));
            const c = (): number => sh.value.x;
            sum += c();
            j++;
        }
    }

    return sum;
}

function main() {
    for (let i = 0; i < 1000; i++) {
        assert(inFinally(4) == 10, "in a finally");
        assert(inCatch(4) == 10, "in a catch clause");
        assert(sharedInFinally(4) == 10, "a Shared in a finally");
    }

    print("done.");
}
