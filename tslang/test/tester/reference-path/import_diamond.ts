// plain_module is imported directly and again through via_module: it is generated once.
import './plain_module'
import './via_module'

assert(plain_fn() == 7);
assert(via_fn() == 8);

print("done.");
