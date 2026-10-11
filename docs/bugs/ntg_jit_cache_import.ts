// With the JIT cache (on by default under --emit=jit), a file that imports another .ts file
// crashes the compiler (0xC0000005) before running, even with an empty cache: only the importing
// file's object is written. With --jit-cache=false it runs. Expected output: 42, done.
import { twice } from "./ntg_jit_cache_import_lib";

const r = twice(21);
print(r);
assert(r == 42);
print("done");
