// A literal of a @value interface is a value: `len2` builds one, nested, and passes it on with no
// allocation in its optimized code (value-no-alloc.cmake reads the IR). So is `new` of a @value
// class: `plus2` makes three and allocates nothing either.
@value
interface Vec {
    readonly x: number;
    readonly y: number;
}

@value
interface Ray {
    readonly start: Vec;
    readonly dir: Vec;
}

function len(r: Ray) {
    return r.start.x + r.dir.y;
}

export function len2(a: number, b: number): number {
    return len({ start: { x: a, y: 0 }, dir: { x: 0, y: b } });
}

@value
class V {
    constructor(public readonly x: number, public readonly y: number, public readonly z: number) {}

    static plus(a: V, b: V) {
        return new V(a.x + b.x, a.y + b.y, a.z + b.z);
    }
}

export function plus2(a: number): number {
    return V.plus(new V(a, 0, 0), new V(a, 0, 0)).x;
}
