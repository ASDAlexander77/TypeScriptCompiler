// A value the builtin cannot cast to what it takes crashed the compiler (#438): the failed cast
// was a null value, which went on to the op's builder.
function main() {
    let h: (() => number)[] = [];
    // a function value, not an arrow written as the item (which takes the element's type)
    const f = () => "x";
    h.push(f);
    print(h.length);
}
