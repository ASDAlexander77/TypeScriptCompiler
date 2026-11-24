// TSLANG_ISSUES 69 (fixed in 180): moved by one move or another on some paths, and on a third path by none: released
// on the edges from the third path.
class Node2 {
    constructor(public kind: int) {}
}

class App {
    a: Node2[] = [];
    b: Node2[] = [];

    pick(o: int) {
        const node = new Node2(o);
        if (o == 1) {
            this.a.push(node);
        } else if (o == 2) {
            this.b.push(node);
        }
    }
}

function main() {
    const app = new App();
    app.pick(1);
    app.pick(2);
    app.pick(3);
    assert(app.a.length == 1 && app.b.length == 1);
    print("done.");
}
