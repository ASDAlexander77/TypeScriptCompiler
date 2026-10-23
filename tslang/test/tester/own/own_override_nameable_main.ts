// The importer of own_err_override_nameable_base: it calls `keep` through `Base`'s vtable, which the
// library's `Der.keep` fills, and reads the kept value after a churn. Correct under every model; under
// own the library is the error (spec 31).
import './own_err_override_nameable_base'
function churn() { const keep: V[] = []; for (let i = 0; i < 2000; i++) keep.push(new V(77777)); return keep.length; }
function put(b: Base) { const a = new V(5); b.keep(a); }
function main() { put(mk()); churn(); assert(first() == 5, "kept value still there"); print("done."); }
