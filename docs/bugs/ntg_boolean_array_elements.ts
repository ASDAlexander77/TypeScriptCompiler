// Elements of a boolean array literal read back wrong: printed inverted, and never true in an
// `if`. Expected output: 2, true, b true, done.
const FLAGS = [false, true, false, true];
let n = 0;
for (let i = 0; i < FLAGS.length; i++) {
    if (FLAGS[i]) {
        n++;
    }
}
print(n);

const b: boolean = FLAGS[1];
print(b);
if (b) {
    print("b true");
}

assert(n == 2);
assert(b);
print("done");
