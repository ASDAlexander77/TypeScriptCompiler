// TSLANG_ISSUES 69 (-mm=own), from native_ts_graphics' examples/nri/extended_dynamic_state2_nri.ts:
// `const node = new SceneNode(...); if (tessellated) this.tessNodes.push(node); else
// this.baselineNodes.push(node);` - a value moved on every path, by a different move on each.

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
    print("done.");
}
