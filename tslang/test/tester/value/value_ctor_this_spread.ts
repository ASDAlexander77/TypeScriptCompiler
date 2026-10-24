// ...nor spread into an object literal (which failed first with an error of its own)
@value
class V {
    readonly a: number;
    readonly b: number;

    constructor() {
        this.a = 1;
        const o = { ...this };
        this.b = o.a;
    }
}

function main() {
    print(new V().b);
}
