// A getter whose body fails, beside a setter, then a read of the property: that error, not a crash (the getter,
// known as the accessor before its body was made, had no function behind it, TSLANG 192).
class O {
    _x: number = 0;
    get x(): number {
        return undefinedName;
    }
    set x(value: number) {
        this._x = value;
    }
}

function main() {
    const n = new O();
    n.x += 1;
    print(n.x);
}
