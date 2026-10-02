// What the builtin map and filter give is a generator, and an iterator has the same builtin
// methods as an array: map, filter, forEach, every, some.
function* count(n: number) {
    for (let i = 1; i <= n; i++) yield i;
}

function main() {
    const a: number[] = [1, 2, 3, 4];

    // a chain in one expression: both calls start at the same place
    let s = 0;
    for (const v of a.map(x => x * 2).map(x => x + 1)) s += v;
    assert(s == 24);

    s = 0;
    for (const v of a.map(x => x * 2).filter(x => x > 2).map(x => x + 1)) s += v;
    assert(s == 21);

    // through a variable
    const m = a.filter(x => x > 1);
    s = 0;
    m.forEach(x => { s += x; });
    assert(s == 9);

    assert(a.map(x => x * 2).every(x => x % 2 == 0));
    assert(!a.map(x => x * 2).every(x => x > 2));
    assert(a.map(x => x * 2).some(x => x > 7));
    assert(!a.filter(x => x > 2).some(x => x > 4));

    // a generator of its own
    s = 0;
    for (const v of count(5).filter(x => x % 2 == 1).map(x => x * 10)) s += v;
    assert(s == 90);

    // the result is still an iterator
    const it = a.map(x => x + 100);
    const first = it.next();
    assert(!first.done && first.value == 101);

    const strs = ["a", "b"];
    let joined = "";
    strs.map(x => x + "_").forEach(x => { joined += x; });
    assert(joined == "a_b_");

    print("done.");
}
