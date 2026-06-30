const a: any = 1;
const b: any = 1;

print(a == b);

assert(a == b);

a = 2;

print(a == b);

assert(a != b);

b = 2;

print(a == b);

assert(a == b);

print("done.");