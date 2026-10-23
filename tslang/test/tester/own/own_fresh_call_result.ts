// A call of a function this module defines hands back a block of the caller's own (spec 34): its
// result points into nothing it was given but what its facts name. So the vector `getPoint(x,
// scene.cam)` makes - a closure's call, a function's, one inside a record given to a call - is not
// a use of the borrow `scene.cam`, and `toDrawing`, which may destroy anything (it builds and calls a
// closure), may come between the call and the vector's last use. The ray tracer's `render` (G8).
class V {
    constructor(public x: number) {}
}

class Cam {
    constructor(public f: V, public r: V) {}
}

interface Scene {
    cam: Cam;
}

interface Ray {
    start: V;
    dir: V;
}

function toDrawing(c: V) {
    let legalize = (d: number) => d > 100 ? 100 : d;
    return { r: legalize(c.x) };
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 50; i++) {
        junk.push(new V(999));
    }

    return junk.length;
}

function direct(cam: Cam, k: number) {
    return new V(cam.f.x + k);
}

class Tracer {
    private trace(ray: Ray) {
        return new V(ray.start.x + ray.dir.x);
    }

    render(scene: Scene, n: number) {
        let sum: number = 0;
        const getPoint = (x: number, cam: Cam) => {
            const recenter = (x: number) => x - n / 2;
            return new V(cam.f.x + cam.r.x * recenter(x));
        };

        for (let x = 0; x < n; x++) {
            const p = getPoint(x, scene.cam);
            const q = direct(scene.cam, x);
            const t = this.trace({ start: scene.cam.f, dir: getPoint(x, scene.cam) });
            const c = toDrawing(p);
            churn();
            sum += c.r + p.x + q.x + t.x;
        }

        return sum;
    }
}

function main() {
    const tracer = new Tracer();
    const sum = tracer.render({ cam: new Cam(new V(3), new V(2)) }, 8);
    print("sum:", sum);
    assert(sum == 124);
    print("done.");
}
