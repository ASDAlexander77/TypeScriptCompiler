// The fields of a @value value are readonly: `++`/`--` assign them too, on a local's value, in a
// method and in a closure inside the constructor.
@value
class V {
    constructor(public readonly x: number) {}

    bump() {
        ++this.x;
        return this.x;
    }
}

@value
class W {
    readonly a: number;

    constructor(a: number) {
        this.a = a;
        const f = () => {
            this.a++;
        };
        f();
    }
}

function main() {
    let v = new V(1);
    v.x--;
    print(v.x, v.bump(), new W(5).a);
}
