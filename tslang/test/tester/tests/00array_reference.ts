// An array is a reference: a change of length through any name is seen through every other (#453).
class Holder { items: number[] = [1]; }
type Rec = { items: number[] };

function pushParam(into: number[]) { into.push(2); }
function pushField(h: Holder) { h.items.push(2); }
function pushRec(r: Rec) { r.items.push(2); }
function grow(into: number[]) { into.push(2); into.push(3); into.push(4); into.push(5); }
function setLen(into: number[]) { into.length = 0; }
function popParam(into: number[]) { into.pop(); }
function setElem(into: number[]) { into[0] = 9; }

function main() {
    const a1: number[] = [1]; setElem(a1); assert(a1[0] == 9, "1 element write through a parameter");
    const a2: number[] = [1]; pushParam(a2); assert(a2.length == 2 && a2[1] == 2, "2 push through a parameter");
    const a3: number[] = [1, 2]; popParam(a3); assert(a3.length == 1, "3 pop through a parameter");
    const a4: number[] = [1, 2]; setLen(a4); assert(a4.length == 0, "4 length= through a parameter");
    const a5: number[] = [1]; const b5 = a5; b5.push(2); assert(a5.length == 2, "5 local alias");
    const h6 = new Holder(); pushField(h6); assert(h6.items.length == 2, "6 class field");
    const h7 = new Holder(); const it7 = h7.items; it7.push(2); assert(h7.items.length == 2, "7 field alias");
    const r8: Rec = { items: [1] }; pushRec(r8); assert(r8.items.length == 2, "8 record field");
    const a9: number[] = [1]; const f9 = () => { a9.push(2); }; f9(); assert(a9.length == 2, "9 closure");
    const aa10: number[][] = [[1]]; aa10[0].push(2); assert(aa10[0].length == 2, "10 nested");
    const aa11: number[][] = [[1]]; const in11 = aa11[0]; in11.push(2); assert(aa11[0].length == 2, "11 nested alias");
    const a12: number[] = [1]; const b12 = a12; assert(a12 === b12, "12 identity");
    const a13: number[] = [7]; grow(a13); assert(a13.length == 5 && a13[0] == 7 && a13[4] == 5, "13 growth");
    const a14: number[] = [1]; const h14 = new Holder(); h14.items = a14; a14.push(2); assert(h14.items.length == 2, "14 stored into a field");

    const a15: number[] = [1]; const x15: any = a15; const y15 = x15 as number[]; y15.push(2); assert(a15.length == 2, "15 through any");
    const c16: number[] = []; const d16: number[] = [];
    assert(!(c16 === d16), "16 two empty arrays are different arrays");
    let truthy = false; if (c16) { truthy = true; } assert(truthy, "16 an empty array is truthy");
    let n16: number[] | null = null; let falsy = true; if (n16) { falsy = false; } assert(falsy, "16 null is falsy");

    print("done.");
}
