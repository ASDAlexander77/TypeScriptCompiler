// Comparing an `any`:
// - with undefined or null was answered at compile time (== false, != true) whatever it held;
// - `null` (and a string) turned into an `any` for the comparison was a pointer, not a box, and
//   its "tag" was read from whatever it pointed at;
// - two strings compared the bytes of their pointers, so equal text built at run time differed;
// - `if (a)` / `!a` of an `any` holding undefined or null threw "Can't cast from any type".
function main() {
    const u: any = undefined;
    assert(u == undefined);
    assert(u === undefined);
    assert(u == null);
    assert(!(u === null));
    assert(!(u != undefined));
    assert(!u);
    if (u) {
        assert(false);
    }

    const n: any = null;
    assert(n == null);
    assert(n === null);
    assert(n == undefined);
    assert(!(n === undefined));
    assert(!n);

    const s: any = "text";
    assert(!(s == undefined));
    assert(!(s == null));
    assert(s != undefined);
    assert(s);

    // a `let` is not folded into its initializer: the comparison boxes "text" itself
    let t: any = "te";
    assert(!(t == "text"));
    t = "text";
    assert(t == "text");
    assert(t === "text");
    assert(!(t != "text"));

    // equal text, built at run time, in two boxes
    const built = "te" + "xt";
    const a: any = built;
    const b: any = "text";
    assert(a == b);
    assert(a === b);
    assert(!(a != b));
    const c: any = "other";
    assert(!(a == c));

    // an `any` holding undefined against another holding null
    assert(u == n);
    assert(!(u === n));

    // many comparisons: the box made for the right-hand side is on the stack
    let hits = 0;
    const words = ["text", "nope"];
    for (let i = 0; i < 100000; i++) {
        if (t == words[i % 2]) {
            hits++;
        }
    }
    assert(hits == 50000);

    print("done.");
}
