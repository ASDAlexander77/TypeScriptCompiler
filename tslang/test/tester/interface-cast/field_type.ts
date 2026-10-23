// T1's `b` is a number, S1's a string: the slot is the string's offset, read as a number.
interface S1 {
    a: number;
    b: string;
}

interface T1 {
    b: number;
}

function main() {
    const s: S1 = { a: 1, b: "x" };
    const t: T1 = s;
    print(t.b);
}
