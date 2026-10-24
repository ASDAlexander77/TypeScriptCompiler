// A @value class is nominal: it is not seen as `any`, which would box a copy without its class.
@value
class V {
    constructor(public readonly x: number) {}
}

function main() {
    const a: any = new V(1);
    print(a);
}
