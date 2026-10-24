// `a?.b.c`: once `?.` meets a missing value the rest of the chain is skipped and the result is
// undefined, as in TypeScript - for a @value type (whose missing value threw at `.dir`) and for a type
// literal (which read 0). Present, the chain reads through; a call at its end is skipped too.
@value
class V {
    constructor(public readonly x: number, public readonly y: number) {}

    len() {
        return this.x + this.y;
    }
}

@value
interface Ray {
    readonly start: V;
    readonly dir: V;
}

@value
interface Hit {
    readonly ray: Ray;
    readonly dist: number;
}

function find(d: number): Hit | null {
    if (d < 0) return null;
    return { ray: { start: new V(d, 1), dir: new V(2, d) }, dist: d };
}

type TV = { x: number; y: number };
type TRay = { start: TV; dir: TV };
type THit = { ray: TRay; dist: number };

function findT(d: number): THit | null {
    if (d < 0) return null;
    return { ray: { start: { x: d, y: 1 }, dir: { x: 2, y: d } }, dist: d };
}

function main() {
    const g = find(-1);
    const r = g?.ray.dir.x;
    print("missing", r === undefined ? "undefined" : r);
    assert(r === undefined);
    assert(g?.ray.dir.len() === undefined);

    const f = find(3);
    print("present", f?.ray.dir.y);
    assert(f?.ray.dir.y === 3);
    assert(f?.ray.dir.len() === 5);

    const tg = findT(-1);
    const tr = tg?.ray.dir.x;
    print("missing literal", tr === undefined ? "undefined" : tr);
    assert(tr === undefined);

    const tf = findT(4);
    assert(tf?.ray.dir.y === 4);

    print("done.");
}
