// A `case` compares as `===` does: two @value values have no identity to compare.
@value
interface R {
    readonly a: number;
}

function main() {
    const r1: R = { a: 1 };
    const r2: R = { a: 1 };
    switch (r1) {
        case r2:
            print("same");
            break;
        default:
            print("other");
    }
}
