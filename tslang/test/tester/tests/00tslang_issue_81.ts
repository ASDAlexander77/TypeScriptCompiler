// TSLANG 81: a value moved on some paths only (`const c = new Ctl(); if (c.read()) this.lights.push(c);`)
// is released on the edges that did not move it.
@final
class Ctl {
    node: int = 0;
    read(ok: boolean): boolean { return ok; }
}
@final
class Model {
    lights: Ctl[] = [];
    load(n: int) {
        for (let i = 0; i < n; i++) {
            const c = new Ctl();
            if (c.read(i != 2)) {
                this.lights.push(c);
            }
        }
    }
    load2(n: int) {
        const c = new Ctl();
        if (c.read(n != 2)) {
            this.lights.push(c);
        }
    }
}
function main() {
    const m = new Model();
    m.load(4);
    m.load2(1);
    m.load2(2);
    assert(m.lights.length == 4, "lights");
    print("done.");
}
