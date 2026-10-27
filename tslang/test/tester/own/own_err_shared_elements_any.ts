// -mm=own rejects: a @sharedElements container of `any`, which may hold an object.
@sharedElements
class Bag<T> {
    items: T[] = [];
}

function main() {
    const bag = new Bag<any>();
    print(bag.items.length);
}
