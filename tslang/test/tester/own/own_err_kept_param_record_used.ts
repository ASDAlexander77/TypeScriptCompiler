// the literal took the kept parameter: reading it afterwards reads a moved value
interface P { k: string; }
function mk(k: string): P { const r: P = { k }; print(k); return r; }
function main() { print(mk("a" + 1).k); }
