// Two imported modules, each referencing common.d.ts.
import './module_left'
import './module_right'

assert(module_left_fn() + module_right_fn() == 30);

print("done.");
