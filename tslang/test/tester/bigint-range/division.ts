// a bigint divided by 0n stops the program, where TypeScript throws a RangeError
function apply(n: bigint, d: bigint) {
    return n / d;
}

function main() {
    print("printed before");
    print("result: ", apply(10n, 0n));
}
