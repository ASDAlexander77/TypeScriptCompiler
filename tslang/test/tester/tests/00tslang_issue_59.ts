// TSLANG_ISSUES 59 (-mm=own), from native_ts_graphics' examples/nri/procedural_geometry_nri.ts: a value moved
// into a field of an object read out of an array, then used - `const queuedFrame = this.queuedFrames[i];
// ... queuedFrame.aabbAttributeBuffer = buffer; const view = buffer ? buffer.createStructuredView(0) :
// null;`. After the store the value is what the field holds (TSLANG 50), whether the object is a local, a
// parameter or an element or field read.

class View {
    n: int;
    constructor(n: int) {
        this.n = n;
    }
}

class Buffer {
    size: int;
    constructor(size: int) {
        this.size = size;
    }

    makeView(): View {
        return new View(this.size * 10);
    }
}

class Frame {
    buffer: Buffer | null = null;
    view: View | null = null;
    views: View[] = [];
}

class Holder {
    frame = new Frame();
    frames: Frame[] = [new Frame(), new Frame()];
}

function makeBuffer(i: int): Buffer | null {
    return i > 0 ? new Buffer(i) : null;
}

export function setup(frames: Frame[]): int {
    let total = 0;
    for (let i = 0; i < frames.length; i++) {
        const frame = frames[i];
        const buffer = makeBuffer(i);
        frame.buffer = buffer;
        const view = buffer ? buffer.makeView() : null;
        frame.view = view;
        if (!view) {
            continue;
        }

        total += view.n;
    }

    return total;
}

export function throughField(h: Holder): int {
    const view = new View(7);
    h.frame.view = view;
    const pushed = new View(8);
    h.frames[1].views.push(pushed);
    return view.n + pushed.n;
}

function main() {
    const frames = [new Frame(), new Frame(), new Frame()];
    assert(setup(frames) == 30);
    assert(frames[0].view == null && frames[2].buffer != null && frames[2].view != null);
    const h = new Holder();
    assert(throughField(h) == 15 && h.frames[1].views.length == 1);
    print("done.");
}
