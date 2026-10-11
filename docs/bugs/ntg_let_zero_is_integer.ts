// `let s = 0` is inferred as an integer, so adding a fraction truncates. In TypeScript `s` is a
// number (double): expected output 1.5, 1.5, 1.5, 3.5, done.
let s = 0;
s += 1.5;
print(s);

let t = 0;
t = t + 1.5;
print(t);

let u: number = 0;
u += 1.5;
print(u);

function sum(values: number[]): number {
    let a = 0;
    for (let i = 0; i < values.length; i++) {
        a += values[i];
    }
    return a;
}
const total = sum([1.5, 0, 2]);
print(total);

assert(s == 1.5);
assert(t == 1.5);
assert(total == 3.5);
print("done");
