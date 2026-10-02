// -mm=own rejects, and does not copy: one string pushed twice in one push. rc's retains for it
// have no single use each, so no copy is made.
function main() {
    const s = "s" + 1;
    const list: string[] = [];
    list.push(s, s);
    print(list[0], s);
}
