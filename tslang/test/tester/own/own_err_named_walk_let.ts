// -mm=own refuses this walk: the head is given up while `node` still borrows the list (the walk itself,
// 00for_optional_class_condition in the corpus, compiles). What this test pins is that the message names
// `node` without --di, where it said 'this value'.
class Node {
    constructor(public value: number, public next: Node | undefined) {}
}

function main() {
    let head: Node | undefined = new Node(1, new Node(2, undefined));
    let sum: number = 0;
    for (let node = head; node; node = node.next) {
        sum += node.value;
        head = undefined;
    }

    print(sum);
}
