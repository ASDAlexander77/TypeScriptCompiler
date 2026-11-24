// TSLANG 183 (114): `arguments.length` counts the arguments the caller gave, as in JavaScript, not the parameters the
// function declares: an optional parameter left out, and one with a default value left out, are not counted.
function h(a: number, b?: number) {
    return arguments.length;
}

function d(a: number, b = 7, c?: string) {
    return arguments.length;
}

function z(a?: number) {
    return arguments.length;
}

function joined(a: number, b?: number) {
    let s = "";
    for (const v of arguments) s += v;
    return s;
}

function sum(a: number, b?: number, c?: number) {
    let t: number = 0;
    for (let i = 0; i < arguments.length; i++) t += arguments[i];
    return t * 10 + arguments.length;
}

function rest(a: number, ...xs: number[]) {
    return arguments.length;
}

function words(a: string, b?: string) {
    let o = "";
    for (const v of arguments) o += v;
    return o;
}

function main() {
    assert(h(1) == 1 && h(1, 2) == 2, "h");
    assert(d(1) == 1 && d(1, 2) == 2 && d(1, 2, "x") == 3, "d");
    assert(z() == 0 && z(5) == 1, "z");
    assert(joined(1) == "1" && joined(1, 2) == "12", "joined");
    assert(sum(1) == 11 && sum(1, 2) == 32 && sum(1, 2, 3) == 63, "sum");
    assert(rest(1) == 1 && rest(1, 2, 3) == 3, "rest");

    // runtime strings, left out at every other call
    let t: number = 0;
    for (let i = 0; i < 2000; i++) t += words("a" + i, i % 2 == 0 ? "b" + i : undefined).length;
    assert(t > 0, "words");

    print("done.");
}
