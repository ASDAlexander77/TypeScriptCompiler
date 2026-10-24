// A @value class is never extended, so @final adds nothing: say it once.
@value
@final
class V {
    constructor(public readonly x: number) {}
}

function main() {
}
