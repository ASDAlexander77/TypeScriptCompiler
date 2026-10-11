// An accumulator started at the integer literal 0 sums only the whole parts: warned.
function sum(values: number[]): number {
    let a = 0;
    for (let i = 0; i < values.length; i++) {
        a += values[i];
    }

    return a;
}

print(sum([1.5, 0, 2]));
