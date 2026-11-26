// TSLANG 204: sweep bugs, continued.
// - a generic function destructuring a parameter of a generic type alias (`function f<TX>({ f }: TG<TX>)`): "Can't
//   resolve property 'f' of type TG<TX>", and with a rest (`{ f, ...rest }: IG<TX>`) "variable 'rest' is referencing
//   generic type" - the pattern was read in the discovery of the function's parameters, before TX is known
// - a field narrowed where it is destructured (`if (o.bar) { const { bar } = o; }`) kept its declared type
// - a function returning `any` given as one returning another type (`(v: any) => any` as `(v: any) => string`)
// - a call tested for a narrowing (`o() === undefined && x`) was made again by the narrowing (twice; three times in
//   an `if`)
// - a `?:` with a branch of no value (`c ? f() : 0`, `f(): void`): "'ts.If' op expects a non-empty block"; it is undefined
type TG<T> = { f: number; g: T; h: string };
interface IG<T> { f: number; g: T; k: number }
// (the rest of an object with a string field is refused under -mm=own: it borrows the field)
type TN<T> = { f: number; g: T; k: number };

function getF<TX>({ f }: TG<TX>) {
    return f;
}

function restLength<TX>({ f, ...rest }: TN<TX>) {
    return rest.k + f;
}

function restOfInterface<TX>({ f, ...rest }: IG<TX>) {
    return rest.k * f;
}

interface O { bar?: string; n: number }

function narrowedLength(o: O) {
    if (o.bar) {
        const { bar } = o;
        const s: string = bar;
        return s.length;
    }

    return -1;
}

function ident(v: any): any {
    return v;
}

function apply(f: (v: any) => string, v: any): string {
    return f(v);
}

function applyN(f: (v: any) => number, v: any): number {
    return f(v);
}

let calls = 0;

function bump(): void {
    calls++;
}

function voidBranches(b: boolean) {
    b ? bump() : 0;
    const y = b ? bump() : 5;
    b ? bump() : bump();
    return y;
}

let made = 0;

function maybe(): number | undefined {
    made++;
    return undefined;
}

function main() {
    assert(getF({ f: 1, g: true, h: "x" }) == 1, "a pattern over a generic alias");
    assert(restLength({ f: 2, g: 1, k: 3 }) == 5, "...with a rest");
    const i: IG<boolean> = { f: 3, g: false, k: 2 };
    assert(restOfInterface(i) == 6, "...of a generic interface");

    assert(narrowedLength({ bar: "abcd", n: 1 }) == 4, "a narrowed field destructured");
    assert(narrowedLength({ n: 2 }) == -1, "...not there");

    const g: (v: any) => string = ident;
    assert(g("a" + 1) == "a1", "an any result read as a string");
    assert(apply(ident, "b") == "b", "...given as an argument");
    assert(applyN(ident, 3.5) == 3.5, "...read as a number");

    assert(voidBranches(true) === undefined && calls == 3, "a branch of no value is undefined");
    assert(voidBranches(false) === 5 && calls == 4, "...the other branch");

    const tested = maybe() === undefined && made > 0;
    assert(tested && made == 1, "a tested call is made once");
    if (maybe() === undefined && made > 0) {
        made += 10;
    }
    assert(made == 12, "...in an if");

    print("done.");
}
