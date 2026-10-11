// The default library's Math.max starts from Number.MIN_VALUE (the smallest positive double)
// rather than -Infinity, so the maximum of negative numbers is 2.22507e-308. Expected output:
// -3, 3, done.
const m = Math.max(-3, -5);
print(m);
assert(m == -3);
const k = Math.min(3, 5);
print(k);
assert(k == 3);
print("done");
