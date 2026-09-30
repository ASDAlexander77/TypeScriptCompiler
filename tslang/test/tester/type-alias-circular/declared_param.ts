// A type alias that contains itself is an error, not a stack overflow (issue #370).
type Node = [value: s32, next: Reference<Node>];

declare function take(n: Reference<Node>): void;

function main() {
    print("ok");
}
