// @keeps and @borrows from another module (export_own_markers.ts): the constructor keeps the tag the
// importer moves in, and the generators borrow the word the importer still owns.
import './export_own_markers'

function main() {
    const h = new Holder(new Tag("kept"));
    assert(h.tag.name == "kept");

    const w = new Word("abc");
    let s = "";
    for (const c of lettersOf(w)) s += c;
    for (const c of w.letters()) s += c;
    assert(s == "abcabc");

    print("done.");
}
