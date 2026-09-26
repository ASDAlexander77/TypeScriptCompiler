// The C entry point's shape: `argv` as a `Ref<string>` is C's `char **` itself (a `string[]` is a
// (data, length) pair, which a C runtime does not pass). The JIT used to refuse any `main` with
// parameters or a result; it now calls it through a thunk that passes the program arguments and
// takes the result as the exit code. The exit code is 0 here, because the runner requires it.
// argv[0] is the program - the input file under the JIT, the executable when compiled - so only
// its presence is checked.
function main(argc: int, args: Ref<string>): int {
    assert(argc >= 1, "argc");
    assert(Deref(args[0]).length > 0, "argv[0]");
    print("done.");
    return 0;
}
