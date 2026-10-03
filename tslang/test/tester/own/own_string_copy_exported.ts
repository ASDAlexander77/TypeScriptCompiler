// An exported class's constructor keeps its string parameter. Its callers cannot be told so (it can
// be called from another module), so they keep their own, and the field takes a copy.
export class Animal {
    name: string;
    constructor(name: string) {
        this.name = name;
    }
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    let n = "cat" + 1;
    const a = new Animal(n);
    n = "dog" + 2;
    assert(churn() == 1000);
    assert(a.name == "cat1" && n == "dog2");

    print("done.");
}
