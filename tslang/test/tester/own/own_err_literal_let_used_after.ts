// -mm=own rejects: `s` moves into the record, and is read again after it.
function make(i: number) {
    let s = "s" + i;
    const o = { s, n: i };
    print(s);
    return o;
}

function main() {
    print(make(1).s);
}
