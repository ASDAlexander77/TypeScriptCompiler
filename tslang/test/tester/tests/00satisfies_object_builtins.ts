// `e satisfies T` is `e`, keeping its own type; a function takes its untyped parameters' types from T.
// `Object.freeze(o)` is `o`; `Object.keys/values/entries(o)` of an object or a tuple are made of its fields.
// A type parameter nothing gives a type is its constraint, else `any`, as an erased instantiation's.
type Fn = (n: number) => number;
interface Named { name: string }

const a = (/*c*/ 10 satisfies number);
const f = ((n) => n * 2) satisfies Fn;
const o = Object.freeze({ a: 1, b: 2 });

function compare<T>(f?: () => T, x?: T) { return f ? 1 : 0; }
function first<T extends Named>(items?: T[]): string { return items ? items[0].name : "none"; }

function main() {
    assert(a == 10 && f(4) == 8);

    const ks = Object.keys(o);
    let s = "";
    for (const k of ks) s += k;
    assert(s == "ab" && ks.length == 2 && o.a == 1);

    const vs = Object.values({ x: 10, y: 20 });
    let t = 0;
    for (const v of vs) t += v;
    assert(t == 30);

    const es = Object.entries({ p: "P", q: "Q" });
    let u = "";
    for (const [k, v] of es) u += k + v;
    assert(u == "pPqQ");

    const more = Object.keys({ c: 0 });
    more.push("d");
    assert(more.length == 2 && more[1] == "d");

    assert(compare((): number => 0) == 1 && compare() == 0 && first() == "none");
    print("done.");
}
