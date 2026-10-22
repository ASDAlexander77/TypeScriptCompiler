// @strict-null false
class W { constructor(public x: number) {} }
class V { items: number[] = [1, 2, 3]; constructor(public w: W) {} get() { return this.w.x; } }
interface I { w: W; }
function mkv(n: number) { return new V(new W(n)); }
function churn(): number { let s: number = 0; const a: V[] = []; for (let i = 0; i < 64; i++) { a.push(mkv(1000 + i)); } for (const e of a) { s += e.w.x; } return s; }
function t(c: boolean) { let v = mkv(1); const m = v.get; v = mkv(2); churn(); return m() + v.w.x; }
function main() { print(t(true), t(false)); }
