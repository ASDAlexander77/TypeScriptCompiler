// TSLANG_ISSUES 69: moved by one move or another on some paths, but on a third path by none, a value
// is still moved on some paths only.
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
    new App().pick(3);
    print("done.");
}
