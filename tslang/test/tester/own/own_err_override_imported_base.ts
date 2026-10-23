// -mm=own rejects: a `public` override of an imported class's abstract method. The library calls it
// through `Base`'s vtable without its facts and releases what it keeps, so it may not keep its
// argument (spec 31) - in an executable or the JIT's entry module too.
import './own_override_base_lib'
class Der extends Base { saved: V[] = []; public keep(v: V) { this.saved.push(v); } }
function churn() { const keep: V[] = []; for (let i = 0; i < 2000; i++) keep.push(new V(77777)); return keep.length; }
function main() { const d = new Der(); run(d); churn(); assert(d.saved[0].x == 5, "kept value still there"); print("done."); }
