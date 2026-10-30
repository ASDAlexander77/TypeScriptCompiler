// A marker names a parameter by an identifier, not a string: an error at the argument, not a crash.
class Item {
    v = 1;
}

let kept: Item[] = [];

@keeps(it)
function store(it: Item) {
    kept.push(it);
}

function main() {
    store(new Item());
    print(kept.length);
}
