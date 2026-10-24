// @strict-null false
// A @value type has no null pointer: `T | null`, `T | undefined` and `v?: T` are an optional
// value. It is returned, passed, held in a field, compared with null and undefined, switched on,
// and read after a check or through `?.`. `T | null | undefined` keeps which of the two it holds.
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

    assert(sw(-1) == 1);
    assert(sw(3) == 2);
    assert(swu([-1]) == 1);
    assert(swu([3]) == 2);

    print("done.");
}
