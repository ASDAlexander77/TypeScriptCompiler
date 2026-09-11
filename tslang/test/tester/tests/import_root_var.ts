// The program half: its root has code, so it has an entry point either way. See
// export_root_var.ts for what this pair is actually testing.

import './export_root_var'

const v = bump();
print(v);

assert(v == 42);

print("done.");
