// -mm=own rejects: `@borrows` on a function that neither makes a generator over its argument nor
// returns one.
class Word {
    constructor(public text: string) {}
}

@borrows("w")
export function copyOf(w: Word) {
    return new Word(w.text + "");
}

function main() {
    const w = new Word("abc");
    print(copyOf(w).text);
}
