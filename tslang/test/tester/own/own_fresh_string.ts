// -mm=own, phase 0: a fresh string per iteration, owned once and destroyed at the end of it.
function main() {
    for (let i = 0; i < 1000000; i++) {
        const s = "item " + i;
        if (i == 999999) print(s);
    }
    print("done.");
}
