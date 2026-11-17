// -mm=own: a parameter the function keeps, put into an object literal it returns (`return { k }` of a
// string `k`): the caller gave the value up, so the literal owns it, as it owns a new object. Also a
// union and an optional field, a link to an earlier record, and an array the function still pushes
// onto through its parameter after the literal took it.
interface P {
    k: string;
    n: number;
}

interface Q {
    k: string | number;
    t: string | undefined;
}

interface Path {
    readonly prev: Path | undefined;
    readonly key: string | number;
}

interface A {
    a: number[];
}

function mk(k: string): P {
    return { k, n: 1 };
}

function mq(k: string | number, t: string | undefined): Q {
    return { k, t };
}

function addPath(prev: Readonly<Path> | undefined, key: string | number): Path {
    return { prev, key };
}

function ma(a: number[]): A {
    const r: A = { a };
    a.push(2);
    return r;
}

function churn() {
    const keep: string[] = [];
    for (let i = 0; i < 100; i++) keep.push("churn" + i);
    return keep.length;
}

function main() {
    let total = 0;
    for (let i = 0; i < 100; i++) {
        const p = mk("a" + i);
        const q = mq("b" + i, "c" + i);
        churn();
        total += p.k.length + p.n + q.t.length;

        let path = addPath(undefined, "root" + i);
        path = addPath(path, i);
        churn();
        total += path.prev.key == "root" + i ? 1 : 0;

        total += ma([i]).a.length;
    }

    assert(total == (290 + 100 + 290) + 100 + 200, "every value read after churn");
    print("done.");
}
