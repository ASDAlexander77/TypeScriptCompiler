// -mm=own: a closure made in a `for (let i ...)` takes the iteration's `i` by value only when the body assigns `i`
// before it (TSLANG 217, 128): here `i++` runs after the closure is made, which JavaScript's closure would see. An
// error, as before.
function main() {
    const fs: (() => number)[] = [];
    for (let i = 0; i < 6; i++) {
        fs.push(() => i);
        i++;
    }

    print(fs[0]());
}
