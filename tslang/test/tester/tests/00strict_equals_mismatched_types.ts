// `===` / `!==` between a boolean, a number and a string never coerce: the values are simply unequal,
// so `===` is false and `!==` is true - for literals, for consts and for values typed `number`
function literals() {
    assert(1 !== true, "1 !== true");
    assert("a" !== 1, "\"a\" !== 1");
    assert(true !== "x", "true !== \"x\"");
    assert(!(1 === true), "1 === true");
    assert(!("a" === 1), "\"a\" === 1");
    assert(!(true === "x"), "true === \"x\"");
}

function consts() {
    const n = 1, b = true, s = "a";
    assert(n !== b, "n !== b");
    assert(s !== n, "s !== n");
    assert(b !== s, "b !== s");
    assert(!(n === b), "n === b");
    assert(!(s === n), "s === n");
    assert(!(b === s), "b === s");
}

function typed() {
    let x: number = 1;
    let b: boolean = true;
    let s: string = "1";
    assert(x !== b, "number !== boolean");
    assert(x !== s, "number !== string");
    assert(s !== x, "string !== number");
    assert(!(x === b), "number === boolean");
    assert(!(x === s), "number === string");

    // loose equality still coerces
    assert(x == b, "1 == true");
}

function main() {
    literals();
    consts();
    typed();
    print("done.");
}
