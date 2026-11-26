// Bugs found by the tsc tests sweep (TSLANG 198):
// - a rest element in an object destructuring assignment (`({ a: x, ...r } = o)`): "rest destructuring in an object
//   destructuring assignment is not supported"; it is an object of the fields the pattern does not name
// - the rest of a `T | undefined` in a declaration (`const { ...rest } = u`): "field ... can't be found in tuple"; T's
//   fields, their defaults where it has no value
function restOfOptional(u: { n: number; m: string } | undefined) {
    const { ...rest } = u;
    const { m, ...rest2 } = u;
    return rest.n + rest2.n;
}

function main() {
    assert(restOfOptional({ n: 2, m: "x" }) == 4 && restOfOptional(undefined) == 0, "the rest of an optional");

    const o = { a: 1, b: "x", c: true };
    let x: number = 0;
    let r: { b: string; c: boolean } = { b: "", c: false };
    ({ a: x, ...r } = o);
    assert(x == 1 && r.b == "x" && r.c, "a rest element in an assignment");

    let y: number = 0;
    let s: { a: number } = { a: 0 };
    ({ b: y, ...s } = { b: 5, a: 7 });
    assert(y == 5 && s.a == 7, "a rest element of an object literal");

    print("done.");
}
