// TSLANG_ISSUES 63-65 (-mm=own), from native_ts_graphics:
// 63: core/nri.ts's lazy getters, `if (!this.cachedVertexBuffer) this.cachedVertexBuffer = new
//     NriBuffer(Nri_GetGltfMeshVertexBuffer(this.handle));` - a fill of an empty field (TSLANG 47) with a C
//     call in between, which cannot reach the field. The getter destroys nothing, so a borrow held across
//     it (`commandBuffer` below) holds.
// 64: examples/nri/simple_pbr_nri.ts, `const queuedFrame = new QueuedFrame(); this.queuedFrames.push(
//     queuedFrame); queuedFrame.pbrView = pbrView; if (!pbrView) ...` - a value moved into a field of an
//     object that was pushed first.
// 65: examples/nri/logic_op_dynamic_state_nri.ts, `let pipeline = null; if (loaded) pipeline = create();
//     this.baselinePipelines[logicOp] = pipeline; return pipeline;` - a value moved into an element, and
//     a local's value moved into a field or an element, then used.

@nodrops declare function abs(n: int): int;

class Buf {
    constructor(public readonly handle: int) {}
}

class Mesh {
    cached: Buf | null = null;
    constructor(public readonly handle: int) {}

    @lends("this")
    getBuffer(): Buf {
        if (!this.cached) {
            this.cached = new Buf(abs(this.handle));
        }

        return this.cached;
    }
}

class Recorder {
    count = 0;
    record(b: Buf) {
        this.count += b.handle;
    }
}

class Ring {
    recorders: Recorder[] = [new Recorder(), new Recorder()];

    @lends("this")
    recorder(i: int): Recorder {
        return this.recorders[i];
    }
}

class View {
    n: int;
    constructor(n: int) {
        this.n = n;
    }
}

class Frame {
    view: View | null = null;
    other: View | null = null;
}

class App {
    ring = new Ring();
    mesh = new Mesh(3);
    frames: Frame[] = [];
    pipelines: (View | null)[] = [null, null, null];
    current: View | null = null;

    draw(): int {
        const recorder = this.ring.recorder(1);
        const mesh = this.mesh;
        recorder.record(mesh.getBuffer());
        recorder.record(mesh.getBuffer());
        return recorder.count;
    }

    addFrame(i: int): int {
        const frame = new Frame();
        this.frames.push(frame);
        const view = i > 0 ? new View(i) : null;
        const other = new View(i + 10);
        frame.view = view;
        frame.other = other;
        if (!view) {
            return -1;
        }

        return view.n + other.n;
    }

    pipeline(index: int, make: boolean): View | null {
        let made: View | null = null;
        if (make) {
            made = new View(index * 100);
        }

        this.pipelines[index] = made;
        return made;
    }

    keep(n: int): int {
        let v = new View(n);
        if (n > 5) {
            v = new View(n * 2);
        }

        this.current = v;
        return v.n;
    }
}

function main() {
    const app = new App();
    assert(app.draw() == 6);
    assert(app.addFrame(0) == -1 && app.addFrame(2) == 14 && app.frames.length == 2);
    const p = app.pipeline(2, true);
    assert(p != null && p.n == 200 && app.pipeline(1, false) == null);
    assert(app.keep(3) == 3 && app.keep(7) == 14);
    print("done.");
}
