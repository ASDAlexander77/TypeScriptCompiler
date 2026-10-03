// Pins a conservative rule: one string pushed twice in one push is rejected under -mm=own, and
// not copied, because rc's two retains for it have no single use each (the `several` guard of
// copyForRetain). With the guard off, the double push compiles and is correct: each retain
// copies a different operand. If the rule is widened, this should become a positive.
function main() {
    const s = "s" + 1;
    const list: string[] = [];
    list.push(s, s);
    print(list[0], s);
}
