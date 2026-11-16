// -mm=own rejects: one `let` given to two fields of a record - the record would own its value twice.
function make(i: number) {
    let s = "s" + i;
    return { a: s, b: s };
}

function main() {
    const o = make(1);
    print(o.a, o.b);
}
