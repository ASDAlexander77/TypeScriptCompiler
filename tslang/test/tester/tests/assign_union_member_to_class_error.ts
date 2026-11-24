// A union assigned to a variable is an error when one of its members goes to none of the variable's
// (TS2322): the string was taken for an N.
class N {
    s = 1;
}

function main() {
    const v: string | null = "a" + 1;
    let n: N | null = null;
    n = v;
    print(n ? n.s : 0);
}
