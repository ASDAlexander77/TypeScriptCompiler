// A generator is its own iterator, as in JavaScript: `g[Symbol.iterator]()` is g, so a generator passes where
// something with [Symbol.iterator] is spelled out. It was "Can't resolve property 'iterator'".
function* count(n: number) {
    for (let i = 1; i <= n; i++) yield i;
}

function main() {
    const it = count(3);
    let s: number = 0;
    for (const v of it[Symbol.iterator]()) s += v;
    assert(s == 6, "a generator's [Symbol.iterator]() is the generator");

    // partly consumed: the same generator, where it was
    const g = count(4);
    g.next();
    let rest: number = 0;
    for (const v of g[Symbol.iterator]()) rest += v;
    assert(rest == 9, "goes on from where it was");

    print("done.");
}
