// `argc` and the exit code as numbers, which lower to double: the JIT's entry thunk converts both.
function main(argc: number): number {
    assert(argc >= 1, "argc");
    print("done.");
    return 0;
}
