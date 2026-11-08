// Objects, arrays and strings made and dropped in loops: what an executable's heap (mimalloc on
// Windows, see mimalloc-heap.cmake) is asked for most.
class Node {
    constructor(public value: number, public label: string) {}
}

function make(i: number): Node {
    return new Node(i, "node-" + i);
}

function churn(rounds: number): number {
    let total: number = 0;
    for (let r = 0; r < rounds; r++) {
        const nodes: Node[] = [];
        for (let i = 0; i < 100; i++) {
            nodes.push(make(i));
        }

        for (const node of nodes) {
            total += node.value + node.label.length;
        }
    }

    return total;
}

function main() {
    const total = churn(200);
    print("total:", total);
    assert(total == 1128000);
    print("done.");
}
