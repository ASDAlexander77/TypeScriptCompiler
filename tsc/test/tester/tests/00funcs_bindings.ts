function objectBindingPattern({ foo }: { foo: number }) {
    print(foo);
    assert(foo == 10.0);
}

function arrayBindingPattern([foo]: number[]) {
    print(foo);
    assert(foo == 1.0);
}

function drawText({ text = "", location: [x, y] = [0, 0], bold = false }) {
    print(text, x, y, bold);
    assert(text == "someText");
    assert(x == 1);
    assert(y == 2);
    //assert(bold);
}

function main() {
    objectBindingPattern({ val: 10.0 });
    arrayBindingPattern([1.0]);

    const item1 = { text: "someText", location: [1, 2, 3], style: "italics" };
    drawText(item1);
    const item2 = { text: "someText", location: [1, 2, 3], bold: true };
    drawText(item2);
        
    print("done.");
}