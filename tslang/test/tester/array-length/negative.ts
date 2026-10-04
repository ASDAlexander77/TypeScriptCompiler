// `length =` with a negative number stops the program (#483)
function shrink(a: number[], n: number) {
    a.length = n;
}

function main() {
    print("printed before");
    const a: number[] = [1, 2, 3];
    shrink(a, -1);
    print("length: ", a.length);
}
