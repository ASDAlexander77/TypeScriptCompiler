// A type alias that contains itself is an error, not a stack overflow (issue #370).
type Node = [value: s32, next: Reference<Node>];

function main() {
    let n: Node;
    print("ok");
}
