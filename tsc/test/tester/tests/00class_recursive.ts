class Node<T> {
    v: T;
    k: string;
    next: Node<T>;

    print() {
    }
}

function main() {
    let n = new Node<number>()
    n.next = n
    n.k = "Hello";
    n.v = 10.0;

    let s = new Node<string>()
    s.next = n
    s.k = "Hello";
    s.v = "rrr";

    print("done.");
}