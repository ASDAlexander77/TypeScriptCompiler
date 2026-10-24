// A @value interface extends only another @value interface: a class's fields are seen through a
// reference, not copied.
class C {
    readonly a: number = 1;
}

@value
interface R extends C {
    readonly b: number;
}

function main() {
}
