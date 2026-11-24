// A union given to a parameter is an error when one of its members goes to none of the parameter's,
// as in TypeScript (TS2345): `string` is assignable to no member of `N | null | undefined`. The
// string's pointer was taken for an N, and reading `n.s` crashed.
class N {
    s = 1;
}

function ok(n: N | null | undefined) {
    return n ? n.s : 0;
}

function main() {
    const v: string | undefined = "a" + 1;
    print(ok(v ?? null));
}
