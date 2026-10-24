// A literal of a @value interface is a value: `len2` builds one, nested, and passes it on with no
// allocation in its optimized code (value-no-alloc.cmake reads the IR).
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
