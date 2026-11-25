// A type stored in a module variable: that error, not "failed to legalize operation 'ts.TypeRef'" (TSLANG 193).
type Foo = number;

export const K = Foo;

function main() {}
