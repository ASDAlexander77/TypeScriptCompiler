// A string made over an array's memory with `<string><Opaque>Ref(buffer[i])` is a view of that
// memory (#481). Under rc it keeps no count; where it is kept - returned, stored, pushed,
// captured - the keeper takes a copy, made after the bytes were written through the view.
declare function strcpy(dst: string, src: string): string;

class Box {
    s: string = "";
}

let g = "";

function conv(value: string): string {
    let buffer: char[] = [];
    buffer.length = 50;
    const s = <string><Opaque>Ref(buffer[0]);
    strcpy(s, value);
    return s;
}

function convLet(value: string): string {
    let buffer: char[] = [];
    buffer.length = 50;
    let s = <string><Opaque>Ref(buffer[0]);
    strcpy(s, value);
    return s;
}

function tail(): string {
    let buffer: char[] = [];
    buffer.length = 50;
    strcpy(<string><Opaque>Ref(buffer[0]), "xyz");
    const s = <string><Opaque>Ref(buffer[1]);
    return s;
}

function keep(b: Box, arr: string[]) {
    let buffer: char[] = [];
    buffer.length = 50;
    const s = <string><Opaque>Ref(buffer[0]);
    strcpy(s, "kept");
    b.s = s;
    g = s;
    arr.push(s);
}

// a view is not a block: the word in front of it is element data, not a count
function frontBytes() {
    let buffer: char[] = [];
    buffer.length = 32;
    strcpy(<string><Opaque>Ref(buffer[0]), "abcdefghijklmnop");
    g = <string><Opaque>Ref(buffer[8]);
    const whole = <string><Opaque>Ref(buffer[0]);
    assert(whole == "abcdefghijklmnop", "the bytes in front of a view are not a count");
    assert(g == "ijklmnop", "a copy of the view");
}

// the array grows after its view was kept: the keeper's copy does not hold the old block
function grows() {
    let buffer: char[] = [];
    buffer.length = 16;
    const s = <string><Opaque>Ref(buffer[0]);
    strcpy(s, "grows");
    g = s;
    for (let j = 0; j < 64; j++) buffer.push(<char>65);
}

function main() {
    const a = conv("1.5");
    const b = convLet("2.25");
    const t = tail();
    const box = new Box();
    const arr: string[] = [];
    keep(box, arr);

    // reuse freed memory: a string left pointing at a freed buffer would show it
    const fill: string[] = [];
    for (let i = 0; i < 4096; i++) fill.push("filler-" + i);

    assert(a == "1.5" && b == "2.25" && t == "yz", "returned");
    assert(box.s == "kept" && g == "kept" && arr[0] == "kept", "kept");

    frontBytes();
    for (let i = 0; i < 3; i++) grows();

    print("done.");
}
