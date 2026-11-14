// TSLANG 78: a value pushed on, or stored into, a field of a local object (`const frame = new Frame()`,
// a value, not a read), then used, is read back from that field.
@final
class Buf { constructor(public id: int) {} }
@final
class Alloc {
    constructor(public id: int) {}
    make(): Buf { return new Buf(this.id); }
}
@final
class Frame {
    allocs: Alloc[] = [];
    bufs: Buf[] = [];
    constructor(public n: int) {}
}
@final
class App {
    frames: Frame[] = [];
    build(): int {
        const frame = new Frame(1);
        for (let f = 0; f < 3; f++) {
            const a = new Alloc(f);
            frame.allocs.push(a);
            frame.bufs.push(a.make());
        }
        this.frames.push(frame);
        return this.frames[0].bufs.length;
    }
}
@final
class Mesh {
    constructor(public n: int) {}
}

@final
class Obj {
    mesh: Mesh | null = null;
}

function fieldStore() {
    const objects: Obj[] = [];
    const o = new Obj();
    const mesh = new Mesh(5);
    o.mesh = mesh;
    assert(mesh.n == 5, "mesh");
    objects.push(o);
}

function main() { assert(new App().build() == 3, "bufs"); fieldStore(); print("done."); }
