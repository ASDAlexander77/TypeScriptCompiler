function objectBindingPattern({ foo }: { foo: number }) {
    print(foo);
    assert(foo == 10.0);
}

function arrayBindingPattern([foo]: number[]) {
    print(foo);
    assert(foo == 1.0);
}

function drawText({ text = "someText", location: [x, y] = [1, 2], bold = true }) {
    print(text, x, y, bold);
    assert(text == "someText");
    assert(x == 1);
    assert(y == 2);
    assert(bold);
}

function main() {
    objectBindingPattern({ foo: 10.0 });
    arrayBindingPattern([1.0]);

    const item1 = { text: "someText", location: [1, 2, 3], style: "italics" };
    drawText(item1);
    
    const item2 = { text: "someText", location: [1, 2, 3], bold: true };
    drawText(item2);

    const item3 = { text: "someText", bold: true };
    drawText(item3);

    const item4 = {};
    drawText(item4);
        
    print("done.");
}