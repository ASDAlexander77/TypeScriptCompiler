// -mm=own: a value read out of an object (`const e = x.s`) stays a borrow of it while the object is merged into
// a conditional's result with null (`c ? x : null`): the merge holds the object as long as it is used, so the
// read is still good after it (the inference follows the object into the merge). See
// own_err_literal_temp_merge_stored for the merge given away first.
class N {
    x: number;
    s: string;
    constructor(x: number) {
        this.x = x;
        this.s = "n" + x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new N(1000 + i));
    }
}
function main() {
    let total: number = 0;
    for (let i = 0; i < 40; i++) {
        const x = mk(i);
        const e = x.s;
        const y: N | null = i % 2 == 0 ? x : null;
        churn();
        total += e.length;
        if (y) {
            total += y.x;
        }
    }

    print(total);
    assert(total == 490, "total");
    print("done.");
}
