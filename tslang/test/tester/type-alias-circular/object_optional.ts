// A type alias that contains itself is an error, not a stack overflow (issue #370).
type Node = { value: number, next?: Node };

function main() {
    const n: Node = { value: 1 };
    print(n.value);
}
