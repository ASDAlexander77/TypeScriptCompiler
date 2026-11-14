// TSLANG_ISSUES 71-72 (-mm=own), from native_ts_graphics:
// 71: examples/nri/raytracing_ao_nri.ts, `const texture = device.createStorageTexture(...); if (!texture)
//     return null; const target = new Target(texture, ...); target.readView = texture.createView(format);` -
//     a value narrowed from `T | null`, kept by a constructor, then used.
// 72: examples/nri/extended_dynamic_state2_nri.ts, `const t: number[] = []; for (...) t.push(...); const
//     node = new SceneNode(..., t, ...); ... this.movingX = t[12];` in a loop - a local kept by a
//     constructor, or pushed, then read; and `let p = make(); ...; this.ps.push(p); return p.k;`.

class View {
    constructor(public n: int) {}
}

class Tex {
    constructor(public n: int) {}
    view(): View {
        return new View(this.n);
    }
}

function load(n: int): Tex | null {
    return n > 0 ? new Tex(n) : null;
}

class Target {
    view: View | null = null;
    constructor(public tex: Tex, public state: int) {}
}

class Node2 {
    constructor(public kind: int, public t: number[]) {}
}

class App {
    targets: Target[] = [];
    nodes: Node2[] = [];
    others: Node2[] = [];
    texs: Tex[] = [];
    first = 0.0;

    target(n: int): Target | null {
        const tex = load(n);
        if (!tex) {
            return null;
        }
        const target = new Target(tex, 1);
        target.view = tex.view();
        this.targets.push(target);
        return target;
    }

    build(n: int) {
        for (let o = 0; o < n; o++) {
            const t: number[] = [];
            for (let i = 0; i < 3; i++) {
                t.push(o * 10 + i);
            }
            const node = new Node2(o % 2, t);
            if (o % 2 == 0) {
                this.nodes.push(node);
            } else {
                if (o == 1) {
                    this.first = t[0];
                }
                this.others.push(node);
            }
        }
    }

    keep(n: int, again: boolean): int {
        let tex = load(n);
        if (again) {
            tex = load(n + 1);
        }
        if (!tex) {
            return -1;
        }
        this.texs.push(tex);
        return tex.n;
    }
}

function main() {
    const app = new App();
    const t = app.target(3);
    assert(t != null && t.view != null && t.view.n == 3 && app.target(0) == null);
    app.build(4);
    assert(app.first == 10.0 && app.nodes.length == 2 && app.others[1].t[2] == 32.0);
    assert(app.keep(2, false) == 2 && app.keep(2, true) == 3 && app.keep(0, false) == -1 && app.texs.length == 2);
    print("done.");
}
