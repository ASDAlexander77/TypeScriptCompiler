class Node { constructor(public value: number, public next: Node | null) {} }

function build(n: number): Node | null {
    let head: Node | null = null;
    for (let i = 0; i < n; i++) head = new Node(i, head);
    return head;
}

function sum(head: Node | null): number {
    let s = 0;
    // A `let` of type `Node | null` is not narrowed by a loop condition (`for (...; p; ...)` fails
    // to verify), so the cast does the narrowing.
    let p = head;
    while (p) {
        const q = p as Node;
        s += q.value;
        p = q.next;
    }
    return s;
}

// A long-lived list that must survive every collection triggered by the garbage below.
const keep = build(1000);
let strings = 0;
for (let round = 0; round < 200; round++) {
    build(5000);                                  // garbage
    const text = "round " + round + " of " + 200; // string garbage
    strings += text.length;
    let arr: number[] = [];                       // `const` arrays reject push()
    for (let i = 0; i < 100; i++) arr.push(i);    // array growth garbage
}
print(sum(keep));   // 499500 - wrong or crashing if any of `keep` was collected
print(strings);
