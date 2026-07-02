let i = -1;

let ui = <uint>i;

let len: uint = 200

let cmp = ui < len;

print(i, ui, <uint>200, cmp, 4294967295 < 200);

assert(cmp == false);

print("done.");