// -mm=own rejects: `twice` says `@sharesElements`, and made for an object element type it would give
// each element a second owner (spec 73). The call is the error; the method itself is stubbed.
class Item {
    constructor(public name: string) {}
}
class Bag<T> {
    items: T[] = [];
    @sharesElements
    twice() {
        const both: T[] = [];
        for (const v of this.items) both.push(v);
        for (const v of this.items) both.push(v);
        return both;
    }
}
function main() {
    const bag = new Bag<Item>();
    print(bag.twice().length);
}
