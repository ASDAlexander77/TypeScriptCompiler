// ...nor a field of an object literal
@value
class V {
    readonly a: number;
    readonly b: number;

    constructor() {
        const o = { v: this };
        this.a = 1;
        this.b = o.v.a + 1;
    }
}

function main() {
    print(new V().b);
}
