type A = { kind: "A", value: number };
type B = { kind: "B", name: string };
type C = { kind: "C", cond: boolean };
type All = A | B | C;
function f(input: All) {
    switch (true) {
        case input.kind === "B":
            if (input.kind === "A") {
                return;
            }
            break;
    }
    input;
}
