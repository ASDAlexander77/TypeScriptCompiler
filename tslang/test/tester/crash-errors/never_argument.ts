type S = { kind: "a", n: number } | { kind: "b", n: number };
function assertNever(x: never): never { throw "unexpected"; }
function f(s: S): number {
    if (s.kind === "a") return 1;
    return assertNever(s);
}
