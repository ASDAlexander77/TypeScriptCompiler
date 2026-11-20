// An object literal whose methods use variables of the function around it, copied to a type with methods: an error.
function run(options: { something(b?: string): void }) { options.something("x"); }
function main() { let n = 0; run({ something(b?: string) { n = 5; } }); }
