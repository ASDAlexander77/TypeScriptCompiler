// -mm=own with the default library (own-defaultlib-facts.cmake): a field lent to a library method
// that frees nothing a caller can reach. The library's facts say so, under the JIT too: they are read
// from the DLL the JIT loads, as its borrow facts are. Without them each call may destroy anything,
// and the field could not be lent.
class Doc {
    constructor(public text: string, public at: Date) {}

    find(word: string) {
        return this.text.indexOf(word);
    }

    time() {
        return this.at.getTime();
    }
}

function main() {
    const d = new Doc("hello world", new Date(301349860000));
    assert(d.find("world") == 6);
    assert(d.time() == 301349860000);
    print("done.");
}
