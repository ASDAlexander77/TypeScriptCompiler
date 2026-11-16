// Under own a WeakRef refers to a class object: its header word holds the weak count.
function main() { const a = [1, 2]; const w = new WeakRef(a); print(a.length); }
