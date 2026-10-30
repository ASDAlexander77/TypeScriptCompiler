// -mm=own copies an array whose elements own nothing (`number[]`) rather than give it a second owner
// (spec 71): an accumulator that starts as an element and is then given new values - reduceRight over
// rows with no initial value - starts from a copy of the element, so the rows keep theirs and the
// result has one owner. A copy of an array can be told from the array, so only a function that says
// `@copies` gets one. Each value is read after churn() has reused anything freed too early.
@copies
function reduceRows<T>(rows: T[], func: (v: T, t: T) => T): T {
    let result = rows[rows.length - 1];
    for (let i = rows.length - 2; i >= 0; i--) result = func(result, rows[i]);
    return result;
}

function join(a: number[], b: number[]) {
    const joined: number[] = [];
    for (const x of a) joined.push(x);
    for (const x of b) joined.push(x);
    return joined;
}

function churn() {
    const hold: number[][] = [];
    for (let i = 0; i < 64; i++) {
        hold.push([i, i + 1, i + 2]);
    }
}

function main() {
    const rows: number[][] = [
        [0, 1],
        [2, 3],
        [4, 5],
    ];

    const joined = reduceRows(rows, join);
    churn();
    assert(joined.length == 6 && joined[0] == 4 && joined[5] == 1, "the rows joined from the right");
    assert(rows[2].length == 2 && rows[2][0] == 4, "the last row is its own");

    // one row: the result is the element itself in JavaScript, and a copy of it under own
    const one: number[][] = [[7, 8]];
    const single = reduceRows(one, join);
    churn();
    assert(single.length == 2 && single[1] == 8, "the one row");
    assert(one[0].length == 2 && one[0][0] == 7, "the row itself");

    print("done.");
}
