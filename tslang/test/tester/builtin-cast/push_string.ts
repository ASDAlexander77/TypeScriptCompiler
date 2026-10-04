// A value the builtin cannot cast to what it takes crashed the compiler (#438): the failed cast
// was a null value, which went on to the op's builder.
function main() {
    let s: string[] = [];
    s.push(() => 1);
    print(s.length);
}
