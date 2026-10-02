// -mm=own, phase 7b rejects: `s` is the caller's, so the generator cannot hand it out as its own.
function* words(s: string) {
    yield s;
}

function main() {
    const s = "w" + 1;
    for (const w of words(s)) {
        print(w);
    }
}
