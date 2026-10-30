// -mm=own rejects: an element taken out of a rest parameter the function does not keep is the
// caller's (spec 72), and `first` would free `item`, which main still reads. It was accepted, and
// the program freed `item` twice.
class Item { constructor(public name: string) {} }
function dropFirst(...items: Item[]) {
    const first = items.shift();
    print(first.name);
}
function main() {
    const item = new Item("kept");
    dropFirst(item);
    print(item.name);
}
