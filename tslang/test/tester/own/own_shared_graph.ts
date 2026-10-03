// Shared<T> (spec 23): a DAG - one item under two parents - and a singly linked list walked with
// `cur = cur.value.next`. Handles are dropped and their memory reused before the rest is read.
class Item {
    v = 0;
    name = "";
    next: Shared<Item> | null = null;
}

class Parent {
    child: Shared<Item> | null = null;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function list(n: number) {
    let head: Shared<Item> | null = null;
    for (let i = 0; i < n; i++) {
        const item = new Shared(new Item());
        item.value.v = i;
        item.value.next = head;
        head = item;
    }

    return head;
}

function main() {
    const child = new Shared(new Item());
    child.value.name = "child" + 1;
    let p1 = new Parent();
    const p2 = new Parent();
    p1.child = child;
    p2.child = child;
    p1 = new Parent();
    assert(churn() == 1000);
    const c = p2.child;
    assert(c !== null && c.value.name == "child1", "the other parent keeps the child");

    let sum = 0;
    let cur = list(10);
    while (cur) {
        sum += cur.value.v;
        cur = cur.value.next;
    }

    assert(sum == 45, "the list walked to its end");
    assert(cur === null);

    print("done.");
}
