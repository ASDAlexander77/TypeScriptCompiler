// -mm=own: a tuple kept in a local, used as an array (`t.map(f)`), would give each element it owns
// a second owner, the array's. A string is copied for the array (TSLANG 215); an array element is not
function main() {
    let t: [number, number[]] = [3, [1, 2]];
    print(t.map(x => 1).length);
}
