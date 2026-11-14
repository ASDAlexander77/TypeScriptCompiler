// TSLANG_ISSUES 53 (-mm=own), from native_ts_graphics' examples: a value of the function's own
// given to a constructor that keeps it in a field, then used.
// - `const sample = new Sample(); const app = new App(sample); sample.add(10);` (each example's main)
// - the object pushed on: `this.frames.push(new Frame(a, b)); a.update(...)` (deferred_shading_nri)
// After the call the value is the object's field, read through the object or the place it was moved
// to last.

class Sample {
    n: int = 0;
    add(k: int) {
        this.n += k;
    }
}

class App {
    items: int[] = [];
    // pushing into another field of its own keeps `sample` where it is
    constructor(private sample: Sample) {
        for (let i = 0; i < 3; i++) {
            this.items.push(i);
        }
    }
    run(): int {
        this.sample.add(1);
        return this.sample.n;
    }
}

class Set1 {
    k: int = 0;
    update(v: int) {
        this.k = v;
    }
}

class Frame {
    constructor(public a: Set1 | null, public b: Set1 | null) {}
}

@final
class Pool {
    alloc(fail: boolean): Set1 | null {
        return fail ? null : new Set1();
    }
}

export class Renderer {
    frames: Frame[] = [];
    init(pool: Pool): boolean {
        for (let i = 0; i < 2; i++) {
            const a = pool.alloc(false);
            const b = pool.alloc(false);
            this.frames.push(new Frame(a, b));
            if (!a || !b) {
                return false;
            }

            a.update(i + 1);
            b.update(i + 10);
        }

        return true;
    }
}

function main() {
    const sample = new Sample();
    const app = new App(sample);
    sample.add(10);
    const n = app.run();
    print(n, sample.n);
    assert(n == 11 && sample.n == 11);

    const r = new Renderer();
    assert(r.init(new Pool()));
    print(r.frames[1].a!.k, r.frames[1].b!.k);
    assert(r.frames[0].a!.k == 1 && r.frames[1].b!.k == 11);
    print("done.");
}
