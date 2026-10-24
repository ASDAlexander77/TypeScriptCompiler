// ...nor in an instance field's initializer, which runs in the constructor: the closure read the copy
// taken before `this.a = 7` (0, not 7)
function mk(v: V) {
    return () => v.a;
}

@value
class V {
    readonly f: () => number = mk(this);
    readonly a: number;

    constructor() {
        this.a = 7;
    }
}

function main() {
    print(new V().f());
}
