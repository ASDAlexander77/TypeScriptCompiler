// A type alias that contains itself is an error, not a stack overflow (issue #370).
export type Node = { value: number, next: Node };

function main() {
    let n: Node;
    print("ok");
}
