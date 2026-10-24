// A @value interface extends only another @value interface: a value has no base layout to widen.
interface I {
    readonly a: number;
}

@value
interface R extends I {
    readonly b: number;
}

function main() {
}
