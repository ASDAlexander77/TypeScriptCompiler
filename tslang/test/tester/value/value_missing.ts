// @strict-null false
// A @value type has no null pointer: `T | null`, `T | undefined`, `T | null | undefined`, `v?: T`
// and `f?: T` are one optional value that keeps which of null and undefined it holds, as TypeScript
// does. It is returned, passed, held in a field, converted between those spellings, compared with
// null and undefined, switched on, and read after a check or through `?.`.
@value
interface Hit {
    readonly dist: number;
}

interface Holder {
    hit?: Hit;
}

class Box {
    hit?: Hit;
}

function find(d: number): Hit | null {
    if (d < 0) return null;
    return { dist: d };
}

function first(xs: number[]): Hit | undefined {
    let best: Hit | undefined = undefined;
    for (const x of xs) {
        const h = find(x);
        if (h != null && (best === undefined || h.dist < best.dist)) best = h;
    }

    return best;
}

function opt(v?: Hit) {
    return v === undefined ? -1 : v.dist;
}

// what an optional parameter was given: nothing, null or a value
function state(v?: Hit) {
    return v === undefined ? 0 : v === null ? 1 : 2;
}

function sw(d: number) {
    switch (find(d)) {
        case null:
            return 1;
        default:
            return 2;
    }
}

function swu(xs: number[]) {
    switch (first(xs)) {
        case undefined:
            return 1;
        default:
            return 2;
    }
}

function main() {
    assert(find(-1) === null);
    assert(!(find(-1) != null));
    assert(find(2).dist == 2);

    assert(first([3, -1, 2]).dist == 2);
    assert(first([-1]) === undefined);

    assert(find(5)?.dist == 5);
    assert(find(-5)?.dist === undefined);

    assert(opt() == -1);
    assert(opt({ dist: 4 }) == 4);

    const none: Holder = {};
    const some: Holder = { hit: { dist: 8 } };
    assert(none.hit === undefined && none.hit?.dist === undefined);
    assert(some.hit !== undefined && some.hit.dist == 8);

    const box = new Box();
    assert(box.hit === undefined);
    box.hit = find(7);
    assert(box.hit.dist == 7);

    let u: Hit | null | undefined = null;
    assert(!(u === undefined));
    assert(u === null);

    // null and undefined stay apart, whichever spelling holds them
    assert(!(find(-1) === undefined));
    assert(!(first([-1]) === null));
    let n: Hit | null | undefined = find(-1);
    assert(n === null && !(n === undefined));
    n = first([-1]);
    assert(n === undefined && !(n === null));
    n = find(9);
    assert(n !== null && n !== undefined && n.dist == 9);
    const asUndefinable: Hit | undefined = find(-1);
    assert(asUndefinable === null);
    const asNullable: Hit | null = first([-1]);
    assert(asNullable === undefined);
    const present: Hit | null = first([5]);
    assert(present.dist == 5);
    assert(find(-1) == null && first([-1]) == null && find(-1) == undefined);

    assert(state() == 0);
    assert(state(null) == 1);
    assert(state({ dist: 1 }) == 2);

    assert(sw(-1) == 1);
    assert(sw(3) == 2);
    assert(swu([-1]) == 1);
    assert(swu([3]) == 2);

    print("done.");
}
