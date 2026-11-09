import './export_class_string_indexer'

function main() {
    const bag = new M.Bag();
    bag["name"] = "Ann";
    bag["city"] = "Oslo";
    bag["name"] = "Bob";

    let key = "city";
    assert(bag["name"] == "Bob");
    assert(bag[key] == "Oslo");

    // a key that names a member is the member
    assert(bag["size"] == 2);

    print("done.");
}
