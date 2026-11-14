// TSLANG_ISSUES 56 (-mm=own), from native_ts_graphics' examples/nri/16bit_arithmetic_nri.ts: a `let`
// given only fields (or parameters) - `let layout = this.computeLayout; if (this.fp16Enabled) layout =
// this.computeLayoutFP16;` - borrows each until the next assignment. A field written once the `let`
// holds another is fine.

class Pipe {
    id: int = 3;
}

class K {
    a: Pipe = new Pipe();
    b: Pipe = new Pipe();
    fp16 = false;

    pick(): int {
        let layout = this.a;
        let other = this.b;
        if (this.fp16) {
            layout = this.b;
            other = this.a;
        }

        return layout.id * 10 + other.id;
    }

    // `a` written after `l` moved on to `b`, in a loop too
    moveOn(p: Pipe): int {
        let l = this.a;
        l = this.b;
        this.a = new Pipe();
        let total = l.id;
        let m = p;
        for (let i = 0; i < 2; i++) {
            total += m.id;
            this.a = new Pipe();
            m = this.b;
        }

        return total;
    }
}

function main() {
    const k = new K();
    k.b.id = 5;
    assert(k.pick() == 35);
    k.fp16 = true;
    assert(k.pick() == 53);
    const n = k.moveOn(new Pipe());
    print(n);
    assert(n == 5 + 3 + 5);
    print("done.");
}
