// A type alias that contains itself is an error, not a stack overflow (issue #370).
namespace N {
    export type Node = { v: number, next: Node };
}

function main() {
    let n: N.Node;
    print("ok");
}
