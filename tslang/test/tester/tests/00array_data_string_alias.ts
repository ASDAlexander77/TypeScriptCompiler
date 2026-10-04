// A string made over an array's data block keeps the block alive after the array is released,
// as the default library's convertNumber does (`<string><Opaque>Ref(buffer[0])`): the header owns
// a reference to the data block, it does not own the block outright (#453).
function text(): string {
    let buffer: char[] = [];
    buffer.length = 40;
    buffer[0] = <char>111;
    buffer[1] = <char>107;
    const s = <string><Opaque>Ref(buffer[0]);
    return s;
}

function main() {
    const s = text();
    // reuse the memory a wrongly freed block would leave behind
    for (let i = 0; i < 4; i++) {
        let filler: char[] = [];
        filler.length = 40;
        for (let j = 0; j < 40; j++) filler[j] = <char>88;
        print(filler.length);
    }
    print(s);
    assert(s == "ok", "a string over an array's data block outlives the array");
    print("done.");
}
