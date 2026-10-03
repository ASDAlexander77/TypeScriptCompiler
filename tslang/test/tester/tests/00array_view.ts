let arr = []

arr = [1, 2, 3, 4, 5];

const arr2 = arr.view(1, 3);
print(arr2.length, arr2[0], arr2[1], arr2[2]);

assert(arr2.length === 3);
assert(arr2[0] === 2);
assert(arr2[2] === 4);

// a view is a copy that holds its own reference to each element: releasing it (it goes out of
// scope in the called function) must leave the source's elements alive. Filler allocations of
// the same size would reuse a block freed by mistake.
class Item { v: number; constructor(v: number) { this.v = v; } }

function peekItem(a: Item[]) {
    // `view(from, to)` includes `to`: two elements
    const w = a.view(0, 1);
    assert(w.length == 2, "view(0, 1) has two elements");
    return w[0].v + w[1].v;
}

function peekText(a: string[]) {
    const w = a.view(1, 1);
    assert(w.length == 1, "view(1, 1) has one element");
    return w[0].length;
}

function viewsReleased() {
    const items: Item[] = [new Item(7), new Item(8)];
    assert(peekItem(items) == 15, "a view of class elements");
    const keepItems: Item[] = [];
    for (let i = 0; i < 64; i++) { keepItems.push(new Item(100 + i)); }
    assert(items[0].v == 7 && items[1].v == 8, "class elements alive after their view was released");

    const texts: string[] = [];
    for (let i = 0; i < 3; i++) { texts.push("text-" + i); }
    assert(peekText(texts) == 6, "a view of string elements");
    const keepTexts: string[] = [];
    for (let i = 0; i < 64; i++) { keepTexts.push("fill-" + i); }
    assert(texts[1] == "text-1" && texts[2] == "text-2", "string elements alive after their view was released");
}

viewsReleased();

print("done.");