// A global's initializer runs in the global's own region, which no function's inference sees.
// What it makes moves into the global, a root nothing gives back: an object literal seen as an
// interface, an intersection, a nested array literal (mutated later), an object with a method.
interface Counter {
    count: number;
    name: string;
}

interface Named {
    label: string;
}

interface Sized {
    size: number;
}

const counter: Counter = { count: 3, name: "c" };
const both: Named & Sized = { label: "b", size: 2 };
const nested = [[1, 2], [3]];
const withMethod = { n: 4, twice() { return this.n * 2; } };

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    assert(churn() == 1000);
    assert(counter.count == 3 && counter.name == "c");
    assert(both.label == "b" && both.size == 2);
    assert(nested.length == 2 && nested[0][1] == 2 && nested[1][0] == 3);

    nested[1].push(4);
    nested.push([5]);
    assert(churn() == 1000);
    assert(nested[1].length == 2 && nested[1][1] == 4 && nested[2][0] == 5);

    assert(withMethod.twice() == 8);
    print("done.");
}
