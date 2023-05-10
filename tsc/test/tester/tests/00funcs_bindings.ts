function objectBindingPattern({ foo }: { foo: number }) {
    print(foo);
    assert(foo == 10.0);
}

function arrayBindingPattern([foo]: number[]) {
    print(foo);
    assert(foo == 1.0);
}

// TODO: optional binding is not finished
/*
function drawText({ text = "", location: [x, y] = [0, 0], bold = false }) {
    print(text, x, y, bold);
    assert(text == "someText");
    assert(x == 1);
    assert(y == 2);
    //assert(bold);
}
*/

// TODO: optional binding is not finished - setting default value is not finished
function drawText2({ text = "", bold = true }) {
    print(<string>text, <boolean>bold);
}

function main() {
    objectBindingPattern({ foo: 10.0 });
    arrayBindingPattern([1.0]);

    /*
    const item1 = { text: "someText", location: [1, 2, 3], style: "italics" };
    drawText(item1);
    const item2 = { text: "someText", location: [1, 2, 3], bold: true };
    drawText(item2);
    */

    const item3 = { text: "someText", bold: true };
    drawText2(item3);

    const item4 = { text: "someText" };
    drawText2(item4);

        
    print("done.");
}