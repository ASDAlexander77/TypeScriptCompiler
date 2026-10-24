// A setter would change a value through one of its copies.
@value
class V {
    constructor(public readonly x: number) {}

    set y(v: number) {
    }
}

function main() {
}
