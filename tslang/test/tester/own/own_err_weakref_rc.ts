// rc's header word is the strong count: a weak count needs a second word, which rc does not have yet.
class C { v = 1; }

function main() { const c = new C(); const w = new WeakRef(c); print(c.v); }
