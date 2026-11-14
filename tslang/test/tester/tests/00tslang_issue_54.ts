// TSLANG_ISSUES 54 (-mm=own), from native_ts_graphics' core/nri_donut_textures.ts: a value merged
// from a `?:` whose branches each bring a fresh value or null - `const pipeline = loaded ?
// device.createPipeline(...) : null` - is the function's own, so it can be moved (pushed, stored)
// like the call's result itself, and used after the move as 50 and 52 follow it.

class Pipe {
    id: int = 3;
}

function make(n: int): Pipe {
    const p = new Pipe();
    p.id = n;
    return p;
}

@final
class Device {
    make(): Pipe | null {
        return new Pipe();
    }
}

export class K {
    pipes: Pipe[] = [];
    layout: Pipe | null = null;

    // pushed
    load(device: Device, loaded: boolean): int {
        const pipeline = loaded ? device.make() : null;
        if (!pipeline) {
            return -1;
        }

        const id = pipeline.id;
        this.pipes.push(pipeline);
        return id;
    }

    // in a loop's body: each iteration merges and moves its own value
    loadAll(device: Device, loaded: boolean): int {
        for (let i = 0; i < 2; i++) {
            const pipeline = loaded ? device.make() : null;
            if (!pipeline) {
                return -1;
            }

            this.pipes.push(pipeline);
        }

        return this.pipes.length;
    }

    // stored into a field and used after, through it
    keep(ok: boolean): int {
        const layout = ok ? make(5) : null;
        this.layout = layout;
        if (!layout) {
            return -1;
        }

        return layout.id;
    }

    // two fresh branches, and one nested in another
    pick(c: boolean, d: boolean): int {
        const p = c ? make(1) : d ? make(2) : null;
        this.layout = p;
        const q = c ? make(10) : make(20);
        this.pipes.push(q);
        return (this.layout ? this.layout.id : 0) + q.id;
    }
}

function main() {
    const k = new K();
    assert(k.load(new Device(), true) == 3 && k.load(new Device(), false) == -1 && k.pipes.length == 1);
    assert(k.loadAll(new Device(), true) == 3 && k.loadAll(new Device(), false) == -1);
    assert(k.keep(true) == 5 && k.keep(false) == -1);
    const a = k.pick(true, false);
    const b = k.pick(false, true);
    const c = k.pick(false, false);
    print(a, b, c);
    assert(a == 11 && b == 22 && c == 20);
    print("done.");
}
