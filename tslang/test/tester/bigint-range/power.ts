// a bigint raised to a negative power stops the program, where TypeScript throws a RangeError
function apply(n: bigint, d: bigint) {
    return n ** d;
}

function main() {
    print("printed before");
    print("result: ", apply(2n, -1n));
}
