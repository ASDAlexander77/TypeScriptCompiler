// The importer of a module with top-level statements (#447): the module's statements have run,
// once, by the time main does.
import "./toplevel_module";

function main() {
    assert(startCount() == 1, "the module's top level ran once, before main");
    assert(twice(21) == 42, "its function");
    print("done.");
}
