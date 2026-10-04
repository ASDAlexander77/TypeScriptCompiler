// The importer of a module with an exported const and a top-level statement (#448).
import "./export_const_module";

function main() {
    assert(k == 5, "the module's exported const");
    print("done.");
}
