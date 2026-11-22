// A `string | number | undefined` (a union that may be missing) put into a string: "undefined" when it has
// none, as JavaScript gives - print, `"" + u`, `u + "!"`, a template - not its union's default ("0").

function text(u: string | number | undefined) {
    return "" + u;
}

function main() {
    let u: string | number | undefined = undefined;
    assert(text(u) == "undefined", "'' + u");
    assert(u + "!" == "undefined!", "u + '!'");
    assert(`${u}` == "undefined", "template");
    print(u);

    let w: string | number | undefined = 5;
    assert("" + w == "5" && w + "!" == "5!", "a number");
    w = "s";
    assert("" + w == "s" && `${w}` == "s", "a string");

    let n: number | undefined = undefined;
    assert("" + n == "undefined", "an optional number");
    let s: string | undefined = undefined;
    assert("a" + s == "aundefined", "an optional string");

    print("done.");
}
