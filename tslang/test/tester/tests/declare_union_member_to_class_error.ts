// A typed declaration given a union is an error when one of its members goes to none of the type's
// (TS2322).
class N {
    s = 1;
}

function main() {
    const v: number | null = 1;
    const n: N | null = v;
    print(n ? n.s : 0);
}
