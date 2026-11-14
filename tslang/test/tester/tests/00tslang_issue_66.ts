// TSLANG_ISSUES 66 (-mm=own), from native_ts_graphics' core/donut_first_person_camera.ts:
// `let dir = this.dir; if (dragging) dir = rotateAround(dir, ...); this.dir = normalize(dir);` - a `let`
// borrowing a field, then given a fresh value on some paths. The fresh values go into a slot of their own,
// released where the `let` is; the `let` borrows them as it borrows the field.

function rot(v: number[], a: number) {
    return [v[0] * a, v[1], v[2]];
}

class Box {
    constructor(public n: int) {}
}

function grow(b: Box) {
    return new Box(b.n + 1);
}

class Cam {
    dir: number[] = [1.0, 0.0, 1.0];
    box = new Box(1);
    loop(n: int) {
        let dir = this.dir;
        for (let i = 0; i < n; i++) {
            dir = rot(dir, 2.0);
        }
        this.dir = rot(dir, 1.0);
    }
    inner(n: int): number {
        let total = 0.0;
        for (let i = 0; i < n; i++) {
            let d = this.dir;
            if (i % 2 == 0) {
                d = rot(d, 3.0);
            }
            total += d[0];
        }
        return total;
    }
    boxed(c: boolean): int {
        let b = this.box;
        if (c) {
            b = grow(b);
        }
        return b.n;
    }
    thrown(c: boolean): int {
        let b = this.box;
        try {
            if (c) {
                b = grow(b);
                throw "x";
            }
        } catch (e) {
            return b.n * 10;
        }
        return b.n;
    }
}

function main() {
    const c = new Cam();
    c.loop(3);
    assert(c.dir[0] == 8.0);
    assert(c.inner(3) == 8.0 * 3 + 8.0 + 8.0 * 3);
    assert(c.boxed(true) == 2 && c.boxed(false) == 1);
    assert(c.thrown(true) == 20 && c.thrown(false) == 1);
    print("done.");
}
