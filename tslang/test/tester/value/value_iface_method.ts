// A @value interface has only fields: a value has no vtable to call a method through.
@value
interface R {
    readonly f: number;
    m(): int;
}

function main() {
}
