// -mm=own rejects: an override (not `public`) of a method of an imported class with a body. The
// library calls it through the class's vtable without its facts, so it may not keep its argument
// (spec 31).
import './own_override_base_lib'
class Der extends Concrete { saved: V[] = []; keep(v: V) { this.saved.push(v); } }
function churn() { const keep: V[] = []; for (let i = 0; i < 2000; i++) keep.push(new V(77777)); return keep.length; }
function main() { const d = new Der(); runConcrete(d); churn(); assert(d.saved[0].x == 5, "kept value still there"); print("done."); }
