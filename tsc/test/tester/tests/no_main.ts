const arr = [1, 2, 3, 4, 5];

let a = 10;

a += 20;

const b = a;

print(a, b);

for (const b of arr)
    print(b);

assert(arr.length == 5);
assert(a == b);
assert(a == 30);

print("done.");