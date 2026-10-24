// ...nor what a closure in the constructor gives back
@value
class V {
    readonly a: number;
    readonly b: number;

    constructor() {
        const get = () => this;
        this.a = 1;
        this.b = get().a + 1;
    }
}

function main() {
    print(new V().b);
}
