// -mm=own --own-fallback=count: a list walked from a local head (`for (let node = head; node; node =
// node.next)`, spec 94). Own repairs the `let` - it borrows each value it is given - and accepts it, so
// the flag must decide it the same way: nothing counted, no warning (00for_optional_class_condition,
// which counted 'node' before).
class Node {
    constructor(public value: number, public next: Node | undefined) {}
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 100; i++) {
        let head: Node | undefined = new Node(i, new Node(2, new Node(3, undefined)));
        for (let node = head; node; node = node.next) {
            total += node.value;
        }

        assert(head!.value == i && head!.next!.value == 2, "the walk only borrows");
    }

    print(total);
    print("done.");
}
