// TSLANG 207 (leftovers of 195 and 198), gc, rc and none: under -mm=own the cast of an `any` reads no converted
// layout, and the rest of a tuple holding a string is refused.

// 195: an `any` holding an object literal written with integers (`{ code: 7 }`, s32 fields) cast to an object type
// of numbers: it threw "the object's fields are not those of type"
function castAny() {
    const a: any = { code: 7 };
    const c = <{ code: number }>a;
    assert(c.code + 0.5 == 7.5);

    const m: any = { x: 1, y: 2.5, name: "p" };
    const p = <{ x: number; y: number; name: string }>m;
    assert(p.x / 4 == 0.25 && p.y == 2.5 && p.name == "p");

    const n: any = { x: 1.5 };
    assert((<{ x: number }>n).x == 1.5);

    let caught = false;
    try {
        const bad = <{ x: number }><any>{ y: "s" };
        print(bad.x);
    } catch (e) {
        caught = true;
    }

    assert(caught, "a layout of other fields still throws");
}

// 198: the rest of a `T | null` ("the rest of a value of type 'null | T' can't be taken"), declared and assigned
type T = { a: number; b: string; c: number };

function restOf(u: T | null) {
    const { a, ...rest } = u;
    assert(a == 1 && rest.b == "x" && rest.c == 3);
}

function restAssigned(u: T | null) {
    let a: number = 0;
    let rest = { b: "", c: 0 };
    ({ a, ...rest } = u);
    assert(a == 2 && rest.b == "y" && rest.c == 4);
}

function main() {
    castAny();
    restOf({ a: 1, b: "x", c: 3 });
    restAssigned({ a: 2, b: "y", c: 4 });
    print("done.");
}
