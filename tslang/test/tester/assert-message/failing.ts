// A failing assert shows its message as it is at run time, after what the program printed
// before it: abort flushes nothing, and that output was lost whenever stdout was a pipe.
function main() {
    print("printed before");
    const n = 7;
    assert(n == 3, "n is " + n);
}
