// @strict-null false
// value_missing.ts with a reference in the @value type: `Hit.thing` is an interface, so a Hit is
// null when the object in its `thing` is null, and `Hit | null` needs no tag (the `Mark` below holds a
// class instead). As there, `T | null`, `T | undefined`, `T | null | undefined`, `v?: T` and `f?: T`
// are one optional value that keeps which of null and undefined it holds, as TypeScript does. It is returned, passed, held in a field, converted between those spellings, compared with
// null and undefined, switched on, and read after a check or through `?.`.
interface Thing {
    n: number;
}

class Ball implements Thing {
    constructor(public n: number) {}
}

const ball = new Ball(1);

@value
interface Hit {
    readonly thing: Thing;
    readonly dist: number;
}

class Owner {
    constructor(public id: number) {}
}

@value
interface Mark {
    readonly at: number;
    readonly owner: Owner;
}

function mark(d: number): Mark | null {
    if (d < 0) return null;
    return { at: d, owner: new Owner(d) };
}

interface Holder {
    hit?: Hit;
}

class Box {
    hit?: Hit;
}

class HitBox implements Holder {
    hit?: Hit;
}

interface Based {
    x: number;
}

// `f` is a member of one side only: of the intersection, an optional one
type Either = Based & ({ f: Hit } | { g: number });

function find(d: number): Hit | null {
    if (d < 0) return null;
    return { thing: ball, dist: d };
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
    assert(opt({ thing: ball, dist: 4 }) == 4);

    const none: Holder = {};
    const some: Holder = { hit: { thing: ball, dist: 8 } };
    assert(none.hit === undefined && none.hit?.dist === undefined);
    assert(some.hit !== undefined && some.hit.dist == 8);

    const box = new Box();
    assert(box.hit === undefined);
    box.hit = find(7);
    assert(box.hit.dist == 7);

    // a class's `hit?: Hit` seen through the interface, in each state, and written through it
    const hitBox = new HitBox();
    const viaImplements: Holder = hitBox;
    assert(viaImplements.hit === undefined && !(viaImplements.hit === null));
    hitBox.hit = null;
    assert(viaImplements.hit === null && !(viaImplements.hit === undefined));
    hitBox.hit = find(6);
    assert(viaImplements.hit.dist == 6);
    viaImplements.hit = find(2);
    assert(hitBox.hit.dist == 2);
    viaImplements.hit = null;
    assert(hitBox.hit === null && !(hitBox.hit === undefined));
    viaImplements.hit = undefined;
    assert(hitBox.hit === undefined && !(hitBox.hit === null));

    // ...and of one that does not say it implements it
    const plainBox = new Box();
    const viaShape: Holder = plainBox;
    assert(viaShape.hit === undefined && !(viaShape.hit === null));
    plainBox.hit = null;
    assert(viaShape.hit === null && !(viaShape.hit === undefined));
    plainBox.hit = find(4);
    assert(viaShape.hit.dist == 4);
    viaShape.hit = find(3);
    assert(plainBox.hit.dist == 3);
    viaShape.hit = null;
    assert(plainBox.hit === null);
    viaShape.hit = undefined;
    assert(plainBox.hit === undefined);

    const withF: Either = { x: 1, f: { thing: ball, dist: 9 } };
    assert(withF.f.dist == 9);
    const withG: Either = { x: 2, g: 3 };
    assert(withG.f === undefined && !(withG.f === null));

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
    assert(state({ thing: ball, dist: 1 }) == 2);

    assert(sw(-1) == 1);
    assert(sw(3) == 2);
    assert(swu([-1]) == 1);
    assert(swu([3]) == 2);

    // the reference in a value read back, and a class reference standing for null
    assert(find(3).thing.n == 1);
    assert(mark(-1) === null && mark(-1) == null && !(mark(-1) === undefined));
    assert(mark(4).owner.id == 4 && mark(4).at == 4);
    const marks: (Mark | null)[] = [mark(1), mark(-1), mark(2)];
    let seen = 0;
    for (const m of marks) if (m != null) seen += m.owner.id;
    assert(seen == 3);

    // typeof, and boxed: null as null, a value as an object
    assert(typeof find(-1) == "null" && typeof find(2) == "object");
    const boxedNull: any = find(-1);
    const boxedHit: any = find(2);
    assert(boxedNull === null && typeof boxedNull == "null");
    assert(!(boxedHit === null) && typeof boxedHit == "object");

    print("done.");
}
