// -mm=own rejects: an Opaque only borrows what it points at (spec 92), and nothing else keeps the new
// object - it would be freed under the pointer. Keep the object in a variable first.
class Item {
    id = 1;
}

class State {
    current: Opaque | null = null;
}

function main() {
    const s = new State();
    s.current = <Opaque>new Item();
    print(s.current != null);
}
