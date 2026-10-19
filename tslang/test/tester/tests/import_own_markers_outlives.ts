// An error by design under -mm=own: @borrows crossed the module boundary, so the generator may not
// be kept where it could outlive the word it borrows.
import './export_own_markers'

function main() {
    const w = new Word("abc");
    const kept = [lettersOf(w)];
    print(kept.length);
}
