// TSLANG_ISSUES 69 (-mm=own), from native_ts_graphics' examples/nri/extended_dynamic_state2_nri.ts:
// `const node = new SceneNode(...); if (tessellated) this.tessNodes.push(node); else
// this.baselineNodes.push(node);` - a value moved on every path, by a different move on each.
// 70: examples/nri/ray_tracing_extended_nri.ts, `this.loadTexture(device, FLAME_TEXTURE_PATH)` with
//     `const FLAME_TEXTURE_PATH = "..."` in its namespace, and `loadTexture` keeping the path - a constant
//     global of a string literal's type holds that literal only, given to a call that keeps it.

namespace Textures {
    const PATH = "flame.dds";
    export class Cache {
        names: string[] = [];
        load(path: string): int {
            for (let t = 0; t < this.names.length; t++) {
                if (this.names[t] == path) {
                    return t;
                }
            }
            this.names.push(path);
            return this.names.length - 1;
        }
        run(): int {
            return this.load(PATH) + this.load(PATH) + this.load("x");
        }
    }
}

class Node2 {
    constructor(public kind: int) {}
}

class App {
    a: Node2[] = [];
    b: Node2[] = [];
    x: Node2 | null = null;
    y: Node2 | null = null;

    load(n: int) {
        for (let o = 0; o < n; o++) {
            const node = new Node2(o);
            if (o % 2 == 0) {
                this.a.push(node);
            } else {
                this.b.push(node);
            }
        }
    }

    pick(o: int) {
        const node = new Node2(o);
        if (o == 1) {
            this.a.push(node);
        } else if (o == 2) {
            this.x = node;
        } else {
            this.y = node;
        }
    }
}

function main() {
    const app = new App();
    app.load(5);
    assert(app.a.length == 3 && app.b.length == 2 && app.b[1].kind == 3);
    app.pick(1);
    app.pick(2);
    app.pick(3);
    assert(app.a.length == 4 && app.x != null && app.x.kind == 2 && app.y != null && app.y.kind == 3);
    assert(new Textures.Cache().run() == 1);
    print("done.");
}
