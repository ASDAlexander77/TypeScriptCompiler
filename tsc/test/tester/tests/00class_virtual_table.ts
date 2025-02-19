function isFinite(val: number): boolean {
    return val != Number.POSITIVE_INFINITY && val != Number.NEGATIVE_INFINITY;
}

class Number {
    public static NEGATIVE_INFINITY = -1.0 / 0.0;

    public static POSITIVE_INFINITY = 1.0 / 0.0;

    constructor(private value: number) {
    }

    public isFinite(): boolean {
        return isFinite(this.value);
    }

    public valueOf() {
        return this.value;
    }
}

function main()
{
    const a = new Number(20);
    assert(a.valueOf() == 20.0);
    print("done.");
}