type Shape = { kind: "a", x: number } | { kind: "b", y: string } | { kind: "c" };
function f(s: Shape): number {
    switch (s.kind) {
        case "a": return s.x;
        case "b": return 1;
        default:
            const _exhaustive: never = s;
            return 0;
    }
}
