// A @value class cannot be extended: a value has no vtable to override through.
@value
class V {
    constructor(public readonly x: number) {}
}

class X extends V {}

function main() {
}
