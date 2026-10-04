// The arrays of a tuple literal built at run time can be changed (#486): a constant array in it
// (`[x, [1]]`) is a heap array of its own, not the literal's constant data that push did not
// resolve on, and a `const` of such a tuple has storage, as a `let` has, so `nt[1][1].push(5)`
// has a reference to change the array through and the arrays are counted.
function g(t: [number, number[]]) { t[1].push(9); return t[1].length; }
function main() {
    let x = 3;
    let t = [x, [1]];
    t[1].push(2);
    assert(t[1].length == 2 && t[1][1] == 2, "let, constant array in a run-time tuple");
    const c = [x, [1]];
    c[1].pop();
    assert(c[1].length == 0, "const, constant array in a run-time tuple");
    const nt = [x, [2, [3, 4]]];
    nt[1][1].push(5);
    assert(nt[1][1].length == 3, "const, nested");
    const s = ["a" + x, [x, x]];
    s[1].push(1);
    assert(s[0] == "a3" && s[1].length == 3, "a string and an array");
    assert(g([x, [1, 2]]) == 3, "to a tuple parameter");
    for (let i = 0; i < 3; i++) {
        const e = [i, [i]];
        e[1].push(i);
        assert(e[1].length == 2, "a new array each pass");
    }
    print("done.");
}
