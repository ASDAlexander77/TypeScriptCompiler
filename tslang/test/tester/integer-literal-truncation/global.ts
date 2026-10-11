// `let s = 0` is an s32 (integer literals are), so storing 1.5 drops the fraction: warned.
let s = 0;
s += 1.5;
print(s);
