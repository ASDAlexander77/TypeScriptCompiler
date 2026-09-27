// Box<Tree> is first specialized while its module is imported, during this file's discovery pass.
// Its members went into the discovery module, which is thrown away, and the specialization was
// then taken as done: calls to them here failed to lower ("failed to legalize operation
// 'ts.SymbolCallInternal'"), found in the BrowserLib sources attached to #231.

import { Tree } from "./box_module";

function find(n: Tree, name: string): Tree | null {
    if (n.name == name) {
        return n;
    }

    for (let i = 0; i < n.children.length; i++) {
        const found = find(n.children[i], name);
        if (found !== null) {
            return found;
        }
    }

    return null;
}

function main() {
    const root = new Tree("root");
    const a = new Tree("a");
    root.children.add(a);
    a.children.add(new Tree("b"));

    assert(find(root, "b") !== null);
    assert(find(root, "z") === null);
    assert(root.children[0].name == "a");

    print("done.");
}
