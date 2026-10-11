// Each module the JIT cache compiles has an MLIRContext of its own. The built-in types (i32,
// f32, Opaque, ...) were kept in a static map, built in the first module's context: the next
// module got types from a context that no longer existed, and the compiler crashed (0xC0000005).
import { widen, half } from './embedded_types_module'

const a: i32 = 21;
const b: i64 = widen(a) * 2;
assert(b == 42);

const c: f32 = 5;
assert(half(c) == 2.5);

print("done.");
