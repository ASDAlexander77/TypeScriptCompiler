// A type alias that contains itself is an error, not a stack overflow (issue #370).
type List<T> = { v: T, next: List<T> };

function main() {
    let n: List<number>;
    print("ok");
}
