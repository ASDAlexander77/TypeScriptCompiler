// A type another module's declarations name but never declare is an error, though the importer
// tries each declaration again once the others are in.
import './export_unresolved_iface'

function main() {
    const a = mkA(2);
    print(a.h.k, a.n);
}
