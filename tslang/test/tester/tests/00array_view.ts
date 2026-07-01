let arr = []

arr = [1, 2, 3, 4, 5];

const arr2 = arr.view(1, 3);
print(arr2.length, arr2[0], arr2[1], arr2[2]);

assert(arr2.length === 3);
assert(arr2[0] === 2);
assert(arr2[2] === 4);

print("done.");