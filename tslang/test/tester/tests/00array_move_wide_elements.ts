// shift, unshift and splice move the elements after the changed position with memmove. The move
// was sized by a pointer rather than by the element, so for elements wider than 8 bytes (a tuple
// here) only part of each element moved, and unshift moved one element count too many, writing
// past the end of the block (under -mm=rc the debug heap caught that on free).
function main() {
    let s: [number, string][] = [[1, "a"], [2, "b"], [3, "c"]];
    s.shift();
    assert(s.length == 2, "shift length");
    assert(s[0][0] == 2 && s[0][1] == "b", "shift first");
    assert(s[1][0] == 3 && s[1][1] == "c", "shift second");

    let u: [number, string][] = [[2, "b"], [3, "c"]];
    u.unshift([1, "a"]);
    assert(u.length == 3, "unshift length");
    assert(u[0][0] == 1 && u[0][1] == "a", "unshift first");
    assert(u[1][0] == 2 && u[1][1] == "b", "unshift second");
    assert(u[2][0] == 3 && u[2][1] == "c", "unshift third");

    let g: [number, string][] = [[1, "a"], [4, "d"]];
    g.splice(1, 0, [2, "b"], [3, "c"]);
    assert(g.length == 4, "splice grow length");
    assert(g[1][0] == 2 && g[2][1] == "c" && g[3][0] == 4 && g[3][1] == "d", "splice grow");

    let k: [number, string][] = [[1, "a"], [2, "b"], [3, "c"], [4, "d"]];
    k.splice(1, 2);
    assert(k.length == 2, "splice shrink length");
    assert(k[0][0] == 1 && k[1][0] == 4 && k[1][1] == "d", "splice shrink");

    print("done.");
}
