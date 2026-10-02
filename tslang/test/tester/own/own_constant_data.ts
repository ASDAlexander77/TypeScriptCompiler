// Data nothing owns needs no owner: a number or null in a union, a function, string literals, a
// tuple of literals. rc retains each where a value of its type could hold a block; own takes
// nothing for them. Compiled with --opt by the runner, where CSE merges equal literals.
type Account = [number, string, string, string?];

function twice(x: number) {
    return x * 2;
}

function apply(f: (x: number) => number, v: number) {
    return f(v);
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    // a union made from a number, then from null
    let n: number | null = 5;
    let m = n;
    let k: number | null = null;
    assert(m == 5 && n == 5 && k == null);

    // functions as values
    let f = twice;
    const g = (x: number) => x + 1;
    assert(apply(f, 3) == 6 && apply(g, 3) == 4);

    // tuples of literals: built, pushed, unshifted, spliced, destructured
    let pairs: [number, string][] = [[1, "a"], [2, "b"]];
    pairs.push([3, "c"]);
    pairs.unshift([0, "z"]);
    pairs.splice(2, 0, [9, "y"]);
    assert(churn() == 1000);
    assert(pairs.length == 5 && pairs[0][1] == "z" && pairs[2][1] == "y" && pairs[4][1] == "c");

    let [num, str] = [1, "foo"];
    assert(num == 1 && str == "foo");

    // equal literals in two tuples, the optional field left out
    const staff: Account[] = [
        [0, "A", "a@"],
        [1, "B", "b@"],
        [2, "C", "c@", "Lead"],
    ];
    let count = 0;
    for (const s of staff) count += s[3] ? 1 : 0;
    assert(count == 1 && staff[2][3] == "Lead");

    print("done.");
}
