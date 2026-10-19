// An error by design under -mm=own: @keeps crossed the module boundary, so the tag is moved into
// the Holder and cannot be used after. Without the marker the importer would borrow it, and both
// the Holder and the importer would free it.
import './export_own_markers'

function main() {
    const t = new Tag("kept");
    const h = new Holder(t);
    print(t.name, h.tag.name);
}
