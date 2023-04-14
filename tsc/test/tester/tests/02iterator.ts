let run1 = false;
let run2 = false;
let test = true;

function main() {
    let arr = [1, 2, 3];

    const it = (function* () { for (const v of arr) yield ((x: typeof v) => {run1 = true; return x + 1; })(v); })();

    for (const v of it) { print(v); if (v > 3 || v < 1) test = false; };

    const it2 = (function* () { for (const v of arr) yield (<T>(x: T) => {run2 = true; return x + 1; })(v); })();

    for (const v of it2) { print(v); if (v > 3 || v < 1) test = false; };

    assert(run1);
    assert(run2);
    assert(test);

    print("done.");
}