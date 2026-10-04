// A value the builtin cannot cast to what it takes crashed the compiler (#438): the failed cast
// was a null value, which went on to the op's builder.
function main() {
    let h: (() => number)[] = [];
    h.unshift(() => 1);
    print(h.length);
}
