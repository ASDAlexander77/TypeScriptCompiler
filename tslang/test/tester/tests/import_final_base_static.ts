// A static import of a @final class (export_final_base.ts): the importer links against the
// import library, by name. A member K inherits from a base its library does not export has no
// symbol there, so only a member of a class defined in this module is called directly.
@static
import './export_final_base'

function main() {
    const k = new K();
    assert(k.km() == 2);
    assert(k.bm() == 1);
    assert(k.s2 == 11);
    print("done.");
}
