// -mm=own rejects: `@borrows` says the result may not outlive `w`, but one path returns a generator
// over another value. The signature can only say one thing for every path.
class Word {
    constructor(public text: string) {}
}

@borrows("w")
export function *lettersOf(w: Word) {
    for (const c of w.text) yield <string>c;
}

const other = new Word("xyz");

@borrows("w")
export function pick(w: Word, first: boolean) {
    if (first) {
        return lettersOf(w);
    }

    return lettersOf(other);
}

function main() {
    const w = new Word("abc");
    for (const c of pick(w, true)) print(c);
}
