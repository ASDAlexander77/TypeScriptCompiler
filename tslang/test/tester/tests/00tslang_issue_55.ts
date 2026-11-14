// TSLANG_ISSUES 55 (-mm=own), from native_ts_graphics: an array literal made only of fields (and
// parameters) and only read - `const directions = [this.dir, this.dir, this.right, ...]` in
// core/donut_first_person_camera.ts, `for (const p of [this.a, this.b]) p.destroy()` - borrows its
// elements while it is used, and gives back only itself. The same field twice is fine.

class Pipe {
    id: int = 3;
    destroy() {
        this.id = 0;
    }
}

class Cam {
    dir: number[] = [0.0, 0.0, 1.0];
    right: number[] = [1.0, 0.0, 0.0];
    up: number[] = [0.0, 1.0, 0.0];

    move(step: number): number[] {
        const m = [0.0, 0.0, 0.0];
        const directions = [this.dir, this.dir, this.right, this.right, this.up, this.up];
        const signs = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0];
        for (let k = 0; k < directions.length; k += 2) {
            for (let i = 0; i < 3; i++) {
                m[i] = m[i] + signs[k] * directions[k][i] * step;
            }
        }

        return m;
    }
}

class K {
    a: Pipe = new Pipe();
    b: Pipe = new Pipe();

    sum(): int {
        const pipes = [this.a, this.b, this.a];
        let total = 0;
        for (let i = 0; i < pipes.length; i++) {
            total += pipes[i].id;
        }

        return total;
    }

    // straight over the literal, a parameter among the fields, a method called on each
    kill(extra: Pipe): int {
        let n = 0;
        for (const p of [this.a, this.b, extra]) {
            n += p.id;
            p.destroy();
        }

        return n + this.a.id;
    }
}

function main() {
    const m = new Cam().move(2.0);
    print(m[0], m[1], m[2]);
    assert(m[0] == -2.0 && m[1] == 2.0 && m[2] == 2.0);

    const k = new K();
    k.b.id = 5;
    assert(k.sum() == 11);
    const extra = new Pipe();
    assert(k.kill(extra) == 11 && k.a.id == 0 && extra.id == 0);
    print("done.");
}
