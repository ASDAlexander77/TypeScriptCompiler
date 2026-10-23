// An interface value cast to another interface that it does not extend - one that only has some of its
// members, in other places - kept its own vtable, which the other interface read by its own slot
// numbers: `HasDist.dist` read `Hit.thing`, its first slot, and `B.g()` called `A.f()`, the method in
// B's slot. The cast now gives the value a vtable laid out for the interface it is cast to.
interface Hit {
    thing: number;
    ray: number;
    dist: number;
}

interface HasDist {
    dist: number;
}

interface RayDist {
    dist: number;
    ray: number;
}

interface Labelled {
    dist: number;
    label?: string;
}

class HitC implements Hit {
    thing: number = 10;
    ray: number = 20;
    dist: number = 30;
}

function mkHit(): Hit {
    return { thing: 1, ray: 2, dist: 3 };
}

function distOf(d: HasDist) {
    return d.dist;
}

interface A {
    f(): number;
    g(): number;
}

interface B {
    g(): number;
}

class AC implements A {
    base = 100;
    f(): number {
        return this.base + 1;
    }
    g(): number {
        return this.base + 2;
    }
}

// a diamond: B2's members come after A2's in AB's vtable
interface A2 {
    a: number;
}

interface B2 {
    b: number;
}

interface AB extends A2, B2 {
    c: number;
}

function main() {
    // an object literal, through a local, a call's result, an argument, `<T>`, an element
    const h: Hit = { thing: 1, ray: 2, dist: 3 };
    const d: HasDist = h;
    assert(d.dist == 3, "a field of an object literal");
    const d2: HasDist = mkHit();
    assert(d2.dist == 3, "a field of a call's result");
    assert(distOf(h) == 3, "a field of an argument");
    assert((<HasDist>h).dist == 3, "a field through <T>");
    const ds: HasDist[] = [h];
    assert(ds[0].dist == 3, "a field of an element");

    // two members in another order
    const rd: RayDist = h;
    assert(rd.dist == 3 && rd.ray == 2, "two fields in another order");

    // a class object
    const hc: Hit = new HitC();
    const dc: HasDist = hc;
    assert(dc.dist == 30, "a field of a class object");
    assert(distOf(hc) == 30, "a field of a class object as an argument");

    // a value already cast, cast again
    const again: HasDist = rd;
    assert(again.dist == 3, "a value cast twice");

    // an optional member the source does not have is absent
    const l: Labelled = h;
    assert(l.dist == 3, "a field next to an absent optional one");
    assert(l.label === undefined, "an absent optional field");

    // methods, of a class object and of an object literal
    const a: A = new AC();
    const b: B = a;
    assert(b.g() == 102, "a method of a class object");
    const o: A = {
        f(): number {
            return 10;
        },
        g(): number {
            return 20;
        }
    };
    const b2: B = o;
    assert(b2.g() == 20, "a method of an object literal");

    // the second interface of a diamond
    const ab: AB = { a: 1, b: 2, c: 3 };
    const b3: B2 = ab;
    assert(b3.b == 2, "the second interface of a diamond");

    // casts made again and again share one vtable per content and keep reading right
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const each: HasDist = i % 2 == 0 ? h : hc;
        sum += each.dist;
    }
    assert(sum == 500 * 3 + 500 * 30, "casts in a loop");

    print("done.");
}
