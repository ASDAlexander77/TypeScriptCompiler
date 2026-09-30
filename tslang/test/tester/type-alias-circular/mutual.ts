// A type alias that contains itself is an error, not a stack overflow (issue #370).
type A = { v: number, b: B };
type B = { w: number, a: A };

function main() {
    let n: A;
    print("ok");
}
