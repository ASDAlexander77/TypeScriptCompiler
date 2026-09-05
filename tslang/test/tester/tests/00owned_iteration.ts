// A `for...of` over the default library's iterator protocol hands the loop variable a reference
// it then takes ownership of. When the elements come from a literal they live in read-only
// constant memory, and when the loop runs out the iterator hands back `undefined` typed as the
// element type - both of which the loop retains before it looks at them.

function sumOfLengths(rows: number[][]): number {
    let total = 0;
    for (const row of rows) {
        total = total + row.length;
    }

    return total;
}

function constantRowsAreIterable(): number {
    const rows = [[1], [2]];

    return sumOfLengths(rows);
}

// Three elements is the interesting width: it is the one whose payload wants more alignment
// than the header in front of it, so a header written naively lands in the wrong place.
function raggedRowsAreIterable(): number {
    const rows = [[1], [2, 3], [4, 5, 6]];
    let total = 0;

    for (const row of rows) {
        for (const cell of row) {
            total = total + cell;
        }
    }

    return total;
}

function constantRowsSurviveTheLoop(): number {
    const rows = [[1, 2], [3, 4]];
    let last = 0;

    for (const row of rows) {
        last = row[1];
    }

    // the rows are still readable after the loop has given every element back
    return last + rows[0][0] + rows[1][1];
}

function emptyRowsIterateNoTimes(): number {
    const rows: number[][] = [];
    let count = 0;

    for (const row of rows) {
        count = count + 1;
    }

    return count;
}

function makeRecord(n: number) {
    return { id: n, label: "r" + n };
}

function recordsAreIterable(): string {
    const records = [makeRecord(1), makeRecord(2)];
    let joined = "";

    for (const record of records) {
        joined = joined + record.label;
    }

    return joined;
}

// A generator's own `{ value, done }` result carries the same `undefined` on its final call, so
// this case reaches it without the default library's iterator in the way.
function* someRecords() {
    yield makeRecord(3);
    yield makeRecord(4);
}

function generatedRecordsAreIterable(): string {
    let joined = "";
    for (const record of someRecords()) {
        joined = joined + record.label;
    }

    return joined;
}

function* labels() {
    yield "a";
    yield "b";
}

function generatedStringsAreIterable(): string {
    let joined = "";
    for (const s of labels()) {
        joined = joined + s;
    }

    return joined;
}

function main() {
    assert(constantRowsAreIterable() == 2, "constant rows");
    assert(raggedRowsAreIterable() == 21, "ragged rows");
    assert(constantRowsSurviveTheLoop() == 9, "rows survive the loop");
    assert(emptyRowsIterateNoTimes() == 0, "empty rows");
    assert(recordsAreIterable() == "r1r2", "records");
    assert(generatedRecordsAreIterable() == "r3r4", "generated records");
    assert(generatedStringsAreIterable() == "ab", "generated strings");

    print("done.");
}
