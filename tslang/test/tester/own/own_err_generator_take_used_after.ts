// A yielded value pushed into an array and read again after: the push does not take it, since the
// read after would be through what the array now owns, so the value still borrows the record.
class C {
    constructor(public x: number) {}
}

function* made(n: int) {
    for (let i = 0; i < n; i++) yield new C(i);
}

function main() {
    const kept: C[] = [];
    for (const c of made(2)) {
        kept.push(c);
        print(c.x);
    }
}
