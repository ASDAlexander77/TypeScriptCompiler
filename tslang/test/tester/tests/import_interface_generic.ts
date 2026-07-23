import './export_interface_generic'

class NumberBox implements M.Box<number> {
    value: number;

    constructor(v: number) {
        this.value = v;
    }

    get(): number {
        return this.value;
    }
}

class NumberStringPair implements M.Pair<number, string> {
    first: number;
    second: string;

    constructor(a: number, b: string) {
        this.first = a;
        this.second = b;
    }

    describe(): string {
        return `${this.second}-${this.first}`;
    }
}

function main() {
    const b: M.Box<number> = new NumberBox(42);
    assert(b.get() == 42);
    assert(b.value == 42);

    const p: M.Pair<number, string> = new NumberStringPair(1, "one");
    assert(p.describe() == "one-1");

    print("done.");
}
