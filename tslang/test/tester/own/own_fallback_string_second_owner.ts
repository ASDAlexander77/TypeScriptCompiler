// -mm=own --own-fallback=count: a string a function keeps, stored in two holders. Own refuses the
// second store; copyForRetain cannot copy it (a parameter the function keeps), so with the flag it is
// counted as an object is: both retains kept, and the parameter released at the exit.
class Holder {
    name: string = "";
}

function link(h: Holder, g: Holder, s: string) {
    h.name = s;
    g.name = s;
}

function main() {
    let n: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        link(h, g, "a" + i);
        n += h.name.length + g.name.length;
    }

    print(n);
    print("done.");
}
