// A generic called with an argument that breaks its constraint warns once, from the final check
// (the inference pass that runs first doesn't report it).
function f<T extends string>(t: T) { return t; }
function main() { print(f(1)); }
