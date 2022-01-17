class Node<T> {
    next: Node<T>;
}

function main() {
    let n = new Node<TypeOf<1>>();
    print("done.");
}