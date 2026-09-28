// A const binding only forbids reassigning the name; the array it holds can still grow and
// shrink. An annotated const (`const a: number[]`) and a const holding a call's result used to
// be bare values, so push/pop/splice/length= had no array to change ("Can't get reference of
// the array").
const g: number[] = [1, 2];

function make(): number[] {
    return [7, 8];
}

function main() {
    const a: number[] = [1, 2];
    a.push(3);
    assert(a.length == 3, "push length");
    assert(a[2] == 3, "push value");

    assert(a.pop() == 3, "pop value");
    a.unshift(0);
    assert(a.length == 3, "unshift length");
    assert(a[0] == 0, "unshift value");

    a.splice(1, 1);
    assert(a.length == 2, "splice length");
    assert(a[1] == 2, "splice value");

    a.length = 1;
    assert(a.length == 1, "length set");

    const c = make();
    c.push(9);
    assert(c.length == 3, "call result length");
    assert(c[2] == 9, "call result value");

    g.push(3);
    assert(g.length == 3, "global length");
    assert(g[2] == 3, "global value");

    print("done.");
}
