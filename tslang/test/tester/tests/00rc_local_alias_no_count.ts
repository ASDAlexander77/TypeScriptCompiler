// -mm=rc: a local alias of a field keeps the array alive only where something could release it.
class Box {
    items: number[] = [1, 2, 3];
}

function replace(b: Box) {
    b.items = [7];
}

// nothing here can release the array: the alias needs no count
function sum(b: Box) {
    const items = b.items;
    let s: number = 0;
    for (let i = 0; i < items.length; i++) {
        s += items[i];
    }

    return s;
}

// `replace` releases the array the alias names: the alias has to hold it
function sumAfterReplace(b: Box) {
    const items = b.items;
    replace(b);
    let s: number = 0;
    for (let i = 0; i < items.length; i++) {
        s += items[i];
    }

    return s + b.items[0];
}

function main() {
    const b = new Box();
    assert(sum(b) == 6, "sum");
    assert(sumAfterReplace(b) == 13, "sum after replace");
    assert(sum(b) == 7, "sum of the new array");
    print("done.");
}
