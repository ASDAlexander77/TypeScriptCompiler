// A union of tuples cast to a longer member fills each element the shorter one lacks (TSLANG 215, USER RULING
// 2026-11-27) - but under strict null checks nothing fills an array: only null could. A compile error.
type Short = [number, string];
type Long = [number, string, number[]];

function third(x: Short | Long) {
    return (<Long>x)[2].length;
}

function main() {
    const x: Short | Long = [1, "a"];
    print(third(x));
}
