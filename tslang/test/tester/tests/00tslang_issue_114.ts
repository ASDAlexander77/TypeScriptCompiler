// TSLANG_ISSUES 114: `arguments` in a function with a rest or a destructured parameter was "not supported".
// A rest parameter's elements are the arguments after the others; a destructured parameter is its argument.
function counted(a: number, ...rest: number[]) { return arguments.length; }
function joined(...parts: string[]) {
    let text = "";
    for (const part of arguments) text += part;
    return text;
}
function pair([a, b]: [number, number], c: number) { return arguments.length; }
function inCondition(a: number, ...rest: number[]) {
    let n = 0;
    for (var i = 0; i < arguments.length; i++) n++;
    return n;
}

function main() {
    assert(counted(1, 2, 3) == 3, "rest: three");
    assert(counted(1) == 1, "rest: none given");
    assert(joined("a" + 1, "b", "c") == "a1bc", "rest of strings");
    assert(inCondition(1, 2, 3) == 3, "in a loop's condition");
    for (let i = 0; i < 3; i++) {
        assert(joined("x" + i) == "x" + i, "in a loop");
    }

    // under -mm=own an object (a tuple here) boxed into `any` is kept by the box, which own refuses
    // for a parameter, destructured or not
    if (__MEMORY_MODEL__ !== "own") {
        assert(pair([1, 2], 3) == 2, "destructured");
    }

    print("done.");
}
