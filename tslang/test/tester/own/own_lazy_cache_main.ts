// TSLANG_ISSUES 47: the getter of own_lazy_cache_lib called twice from another module. Its fill of
// the empty field frees nothing, so the first call's result is still there after the second.
import './own_lazy_cache_lib'
function main() {
    const d = new Nri.Device();
    const a = d.getGraphicsQueue();
    const b = d.getGraphicsQueue();
    print(a.h, b.h, a === b);
    assert(a === b && a.h == 7 && d.made == 1);
    print("done.");
}
