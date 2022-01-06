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
    print("done.");
}