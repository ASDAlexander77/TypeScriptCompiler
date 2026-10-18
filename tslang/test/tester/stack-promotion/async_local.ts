// An async function is a coroutine: no promotion in version 1.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

async function f(i: number) {
    const p = new P(i);
    await 1;
    return p.x;
}

async function main() {
    const v = await f(5);
    assert(v == 5);
    print("done.");
}
