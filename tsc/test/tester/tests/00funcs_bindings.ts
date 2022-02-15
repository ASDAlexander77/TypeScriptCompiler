function objectBindingPattern({ foo }: { foo: number }) {
    print(foo);
    assert(foo == 10.0);
}

function arrayBindingPattern([foo]: number[]) {
    print(foo);
    assert(foo == 1.0);
}

// TODO: finish default values for object fields
function drawText({ text = "", location: [x, y] = [0, 0], bold = false }) {
    print(text, x, y, bold);
    assert(text == "someText");
    assert(x == 1);
    assert(y == 2);
    assert(bold);
}

function main() {
    objectBindingPattern({ val: 10.0 });
    arrayBindingPattern([1.0]);

    // TODO: finish it
    //const item = { text: "someText", location: [1, 2, 3], style: "italics" };
    const item = { text: "someText", location: [1, 2, 3], bold: true };
    drawText(item);
        
    print("done.");
}