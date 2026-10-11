// Escapes in a template literal with substitutions stay unprocessed: withSubstitution below holds
// the four characters 1, backslash, n, 1 (a template without substitutions and a string literal
// are fine). Expected output: 3, 3, 3, done.
const plain = "a\nb";
print(plain.length);
assert(plain.length == 3);

const template = `a\nb`;
print(template.length);
assert(template.length == 3);

const n = 1;
const withSubstitution = `${n}\n${n}`;
print(withSubstitution.length);
assert(withSubstitution.length == 3);

print("done");
