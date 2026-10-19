// An error by design under -mm=own: @lends crossed the module boundary, so the tag tagOf lends is
// not read after the holder that owns it is overwritten.
import './export_own_markers'

function main() {
    let h = new Holder(new Tag("kept"));
    const t = tagOf(h);
    h = new Holder(new Tag("other"));
    print(t.name);
}
