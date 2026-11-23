// TSLANG_ISSUES 133: a rest parameter typed as a tuple (`...args: [string, number]`) could not be called with
// separate arguments ("'ts.CreateArray' op result #0 must be ..."): the arguments now make the tuple, each as
// its element's type, and a tuple spread into the call is passed as it is.
function fn(...args: [string, number]) { return args[0] + args[1]; }
function count(...args: [number, number, number]) { return args.length; }
function parts(prefix: string, ...rest: [boolean, string]) {
    const [flag, text] = rest;
    return prefix + (flag ? text : "-");
}

function main() {
    assert(fn("a", 1) == "a1", "separate arguments");
    const t: [string, number] = ["b", 2];
    assert(fn(...t) == "b2", "a tuple spread in");
    assert(count(1, 2, 3) == 3, "length");
    assert(parts(">", true, "x") == ">x" && parts(">", false, "x") == ">-", "after a fixed parameter");
    for (let i = 0; i < 3; i++) {
        assert(fn("c" + i, i) == "c" + i + i, "in a loop, a string made at run time");
    }

    print("done.");
}
