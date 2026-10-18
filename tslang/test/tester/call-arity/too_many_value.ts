// A function value without @varargs takes no extra arguments.
declare function puts(s: string): s32;

function main() {
    let p = puts;
    p("a", "b");
}
