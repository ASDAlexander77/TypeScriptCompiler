// Two @value values have no identity to compare: each is a copy.
@value
class V {
    constructor(public readonly x: number) {}
}

function main() {
    print(new V(1) === new V(1));
}
