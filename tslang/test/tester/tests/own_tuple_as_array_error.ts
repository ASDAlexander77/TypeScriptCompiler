// -mm=own: a tuple kept in a local, used as an array (`t.map(f)`), would give each element it owns
// a second owner, the array's
function main() {
    let t: [number, string] = [3, "b" + 1];
    print(t.map(x => 1).length);
}
