// @strict-null false
// A missing @value value has no fields: reading one, calling a member of one, or converting one to
// the plain type throws, as a member of a null object does.
@value
interface Hit {
    readonly dist: number;
}

@value
class V {
    constructor(public readonly x: number) {}

    get twice() {
        return this.x * 2;
    }
}

function find(d: number): Hit | null {
    if (d < 0) return null;
    return { dist: d };
}

function first(d: number): Hit | undefined {
    if (d < 0) return undefined;
    return { dist: d };
}

function mk(x: number): V | null {
    if (x < 0) return null;
    return new V(x);
}

function main() {
    let caught = 0;

    try {
        print(find(-1).dist);
    } catch (e) {
        caught++;
    }

    try {
        print(first(-1).dist);
    } catch (e) {
        caught++;
    }

    try {
        print(mk(-1).twice);
    } catch (e) {
        caught++;
    }

    // under `@strict-null false` the conversion to the plain type throws, before any read
    try {
        const h: Hit = find(-1);
        print("not here");
    } catch (e) {
        caught++;
    }

    assert(caught == 4);

    // a value there is read as it is
    assert(find(3).dist == 3 && mk(4).twice == 8);
    const h: Hit = find(5);
    assert(h.dist == 5);

    print("done.");
}
