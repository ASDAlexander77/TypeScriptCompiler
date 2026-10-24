// A @value class has no base: the layout of a value is its own fields.
class B {}

@value
class V extends B {
    constructor(public readonly x: number) {}
}

function main() {
}
