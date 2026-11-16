// -mm=own: a tuple literal inside an object literal (`{ t: [i, [i]] }`) is built in a local of its
// own and read once into the field: it moves into the record whole, with the array it holds, and the
// record's owner frees both.
class C {
    v: number[] = [];
}

function churn() {
    let keep: number[][] = [];
    for (let i = 0; i < 200; i++) keep.push([-i, -i, -i]);
    return keep.length;
}

function tupleField(i: number) {
    const o = { t: [i, [i]], n: i };
    churn();
    return o.t[1].length == 1 && o.t[1][0] == i;
}

function returned(i: number) {
    return { t: [i, [i, i]], c: new C() };
}

function main() {
    let ok = 0;
    for (let i = 0; i < 2000; i++) {
        if (tupleField(i)) ok++;
        const r = returned(i);
        churn();
        if (r.t[1].length == 2 && r.t[1][1] == i && r.c.v.length == 0) ok++;
    }

    assert(ok == 4000, "ok");
    print("done.");
}
