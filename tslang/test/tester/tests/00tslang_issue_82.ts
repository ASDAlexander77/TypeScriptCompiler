// TSLANG 82: a closure that calls a function value it captured (`const f = (y) => ...; const rcp = (y) =>
// f(y)`): every store into the captured `f` is a function's symbol, and no closure writes it, so the
// call is a call of that function, which destroys nothing; the method's callers keep their borrows.
class Holder { n: number = 1.0; }
class P {
    holder: Holder | null = new Holder();
    values: number[] = [2.0];
    calc(h: Holder | null, x: number): number {
        const f = (y: number): number => y * 2;
        const rcp = (y: number): number => {
            const r = f(y);
            return f(r + r);
        };
        return h ? rcp(x * h.n) : 0.0;
    }
    run(): number {
        const a = this.values;
        const r = this.calc(this.holder, 1.0);
        return a[0] + r;
    }
}
function main() {
    assert(new P().run() == 10, "run");
    print("done.");
}
