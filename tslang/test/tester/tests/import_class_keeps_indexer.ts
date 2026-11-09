import './export_class_keeps_indexer'

function main() {
    const bag = new M.Bag();
    let a = "a";
    let b = "b";
    bag[a] = new M.Item(1);
    assert(bag[a].n == 1);
    bag[b] = new M.Item(2);
    bag.set("c", new M.Item(3));

    assert(bag.size == 3);
    assert(bag[b].n == 2);
    assert(bag["c"].n == 3);

    print("done.");
}
