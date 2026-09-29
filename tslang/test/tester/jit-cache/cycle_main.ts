// An import cycle with code at the top level: cycle_module cannot be compiled by itself, so the
// JIT cache compiles the program as one module.
import './cycle_module'

export function fromMain() {
    return "main";
}

assert(fromModule() == "module+main");

print("done.");
