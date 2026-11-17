// A function's `var` is a local of that function (MDN var rules), so -mm=own handles it as it
// handles a `let`: holding what a function returns from its rest parameter is accepted. While every
// `var` was a module global this was refused ('h' borrows the result of 'tag' and cannot be stored).
function tag(...x: string[]): string {
    return x[0];
}

function main() {
    var h = tag("a" + 1, "b");
    print(h);
    assert(h == "a1");
    print("done.");
}
