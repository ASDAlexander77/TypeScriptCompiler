import { Node } from "./cycle_class_node";

function main() {
    const n = new Node();
    n.addText("a");
    n.addText("b");
    const inner = new Node();
    inner.addText("c");
    n.childNodes.push(inner);
    assert(n.textContent == "abc");

    print("done.");
}
