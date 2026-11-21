// Destructuring at the module's level and in assignments:
// - a `var` pattern reads a global declared before it: its constructor ran before that global's initializer
//   (a crash);
// - a `var` pattern whose names were all declared before is an assignment to them ("redefinition of symbol");
// - a default of an assignment target (`{ a: x = 1 } = o`, `[x = 1] = arr`) is the target's default, not an
//   assignment to evaluate; a hole takes nothing; an element past the end of an array is the default.
interface R {
    name: string;
    skill?: string;
}

var rA: R = { name: "mower", skill: "mowing" };
var rB: R = { name: "trimmer" };
var { name: nameA } = rA;
var { name: nameA } = rB;
var [x, , y = 5] = [1, 2];
var [x, , y = 5] = [7, 8, 9];

let s = "";
let t = "";
let a = 0;
let b = 0;
let c = 0;

function main() {
    assert(nameA == "trimmer");
    assert(x == 7 && y == 9);

    ({ skill: s = "none" } = rB);
    assert(s == "none");
    ({ skill: s = "none" } = rA);
    assert(s == "mowing");
    ({ name: t = "x" } = rB);
    assert(t == "trimmer");

    [a, , b = 5] = [1, 2];
    assert(a == 1 && b == 5);
    [a, , b = 5, c = 6] = [7, 8, 9];
    assert(a == 7 && b == 9 && c == 6);
    const arr: number[] = [3, 4];
    [a, , b = 10] = arr;
    assert(a == 3 && b == 10);

    // a target that is a pattern itself is destructured from its element or field
    [[a, b] = [1, 2]] = [[2, 3]];
    assert(a == 2 && b == 3);
    const p: { q: number[]; r: { s: number } } = { q: [6, 7], r: { s: 8 } };
    ({ q: [a, b], r: { s: c } } = p);
    assert(a == 6 && b == 7 && c == 8);

    print("done.");
}
