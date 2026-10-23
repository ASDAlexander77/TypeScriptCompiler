// M's `g` is a method, F's a field holding a function: their vtable slots hold different things (a
// function, an offset into the object), so the one can't stand for the other.
interface F {
    k: number;
    g: () => number;
}

interface M {
    g(): number;
}

function main() {
    const f: F = { k: 1, g: (): number => 5 };
    const m: M = f;
    print(m.g());
}
