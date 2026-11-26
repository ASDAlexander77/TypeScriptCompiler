// Bugs found by the tsc tests sweep (TSLANG 196):
// - `m! && m[0]` of a `string[] | null`: `m!` tested did not narrow `m`, and `m[0]` of the union (one pointer) gave
//   ElementRef the union ("operand #0 must be ... array")
function first(m: string[] | null) {
    return m! && m[0];
}

// - a shape with an optional field given to a union with a member for each case of it: "'ts.Cast' op type ... can't
//   be stored"; whether the field has a value tells which
type Foo = { foo?: undefined } | { foo: number };

function toFoo(x: { foo?: number }): Foo {
    return x;
}

// - an object literal given to a union of shapes, told apart by a field it gives as `undefined`, leaves out, gives a
//   template of literals or a const for, or gives where a member has `never`: the member was not found ("can't be
//   stored"), and its function's parameter had no type
type DT = { disc: true; cb: (x: string) => number };
type DF = { disc?: false; cb: (x: number) => number };

function disc(o: DT | DF) {
    return o.disc === true ? "t" : "f";
}

type S = { d: "s"; cb: (x: string) => string };
type N = { d: "n"; cb: (x: number) => string };

function dOf(o: S | N) {
    return o.d;
}

type ADT = { kind: "a"; method(x: string): number } | { kind: "b"; method(x: number): string };

function kindOf(item: ADT) {
    return item.kind;
}

type Link = { href: string; onClick?: (event: string) => string };
type Button = { href?: never; onClick?: (event: number) => string };

function linkOrButton(p: Link | Button) {
    return p.href !== undefined ? "link" : "button";
}

// - a union of shapes with the same fields tested for truth (`if (u)`, the test of `u?.kind`): they merge into one
//   type, and the cast failed with nothing said ("failed statement")
type U2 = { kind: "a"; data: string } | { kind: 1; data: number };

function kindIsString(obj: U2 | undefined) {
    if (typeof obj?.kind === "string") {
        return "s";
    }

    return "n";
}

function main() {
    assert(first(["x"]) == "x", "m! && m[0]");

    const a = toFoo({ foo: 3 });
    const b = toFoo({});
    assert(a.foo == 3 && b.foo === undefined, "an optional field telling the member");

    assert(disc({ disc: true, cb: s => s.length }) == "t", "disc: true");
    assert(disc({ disc: false, cb: n => n + 1 }) == "f", "disc: false");
    assert(disc({ disc: undefined, cb: n => n + 1 }) == "f", "disc: undefined");
    assert(dOf({ d: `${"s"}`, cb: x => x + "!" }) == "s", "a template of literals");
    const kind = "a";
    assert(kindOf({ kind, method(x) { return x.length; } }) == "a", "a const");
    const p: Link | Button = { href: `2${"x"}3`, onClick: ev => ev + "!" };
    assert(linkOrButton(p) == "link", "a never field");

    assert(kindIsString(undefined) == "n" && kindIsString({ kind: "a", data: "x" }) == "s"
        && kindIsString({ kind: 1, data: 2 }) == "n", "typeof u?.kind");

    print("done.");
}
