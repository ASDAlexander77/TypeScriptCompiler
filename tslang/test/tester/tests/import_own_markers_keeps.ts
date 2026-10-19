// @keeps from another module under -mm=own (export_own_markers.ts), on its own: the importer moves
// the tag into the Holder's constructor, and only the Holder frees it.
import './export_own_markers'

function main() {
    const h = new Holder(new Tag("kept"));
    assert(h.tag.name == "kept");
    print("done.");
}
