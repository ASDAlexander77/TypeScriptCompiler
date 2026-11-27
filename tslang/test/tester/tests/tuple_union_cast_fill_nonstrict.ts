// A union of tuples cast to a longer member without strict null checks (TSLANG 215, USER RULING 2026-11-27): a
// string, a class or an array the shorter member lacks is filled with null. Compiled with --strict-null-checks=false.
class Box {
    constructor(public n: number) {}
}

type Short = [number, boolean];
type Long = [number, boolean, string, Box, number[]];

function main() {
    const x: Short | Long = [1, true];
    const l = <Long>x;
    assert(l[0] == 1 && l[1] && l[2] == null && l[3] == null && l[4] == null);
    print("done.");
}
