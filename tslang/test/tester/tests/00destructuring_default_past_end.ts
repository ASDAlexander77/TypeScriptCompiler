// Array destructuring past the end of its source: an element's default is its value where the element is missing
// (`const [a = 1, b = 2] = [5]`: b is 2, as in TypeScript). It was not applied: the default of the element type was
// used (b was 0, a string null), and with objects (`const [a = mk(1), b = mk(2)] = [mk(5)]`, a typed tuple, a nested
// pattern, a for...of pattern) the compiler crashed. Numbers, strings, a typed array, an undefined element, a
// parameter pattern of numbers; every memory model. (Objects: 00destructuring_default_past_end_objects.ts.)
function sum([a = 1, b = 2]: number[]): number {
    return a + b;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 100; i++) {
        const [a = 1, b = 2] = [i];
        total += a + b;

        const [s = "x", t = "y"] = ["s" + i];
        total += s.length + t.length;

        const xs: number[] = [];
        if (i % 2 == 0) {
            xs.push(i);
        }

        const [p = 1, q = 2] = xs;
        total += p + q;

        const [u = 3] = [undefined];
        total += u;

        const [v, w = "q"] = [i];
        total += v + w.length;

        total += sum([i]) + sum([]);
    }

    print(total);
    assert(total == 19040, "total");
    print("done.");
}
