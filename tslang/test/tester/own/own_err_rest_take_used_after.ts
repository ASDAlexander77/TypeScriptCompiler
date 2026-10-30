// -mm=own rejects: `push` keeps the elements it takes out of its rest parameter (spec 72), so `item`
// moves into the call, and main reads it after. It was accepted, and the program freed `item` twice.
class Item { constructor(public name: string) {} }
class List {
    data: Item[] = [];
    push(...args: Item[]) {
        while (args.length > 0) this.data.push(args.shift());
    }
}
function main() {
    const list = new List();
    const item = new Item("a");
    list.push(item);
    print(item.name);
}
