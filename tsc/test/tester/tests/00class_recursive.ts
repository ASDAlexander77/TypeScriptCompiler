class Node<T> {
    v: T;
    k: string;
    next: Node<T>;
}

function main() {
    let n = new Node<number>()
    n.next = n
    n.k = "Hello";
    n.v = 10.0;

    let s = new Node<string>()
    // TODO: bug, allow to set Node<number> to Node<string>
    //s.next = n
    s.next = s
    s.k = "Hello";
    s.v = "rrr";

    print("done.");
}