// ...nor an argument
function show(v: V) {
    return v.a;
}

@value
class V {
    readonly a: number;
    readonly b: number;

    constructor() {
        this.a = 1;
        this.b = show(this);
    }
}

function main() {
    print(new V().b);
}
