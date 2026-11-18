type S = { kind: "a", n: number } | { kind: "b", n: number };
function f(s: S) {
    if (s.kind === "a") return;
    if (s.kind === "b") return;
    s();
}
