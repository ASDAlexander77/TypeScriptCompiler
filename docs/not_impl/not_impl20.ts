function drawText({ text = "", location: [x, y] = [0, 0], bold = false }) {
    print(text, x, y, bold);
    assert(text == "someText");
    assert(x == 1);
    assert(y == 2);
    assert(bold);
}

function main() {

    //const item = { text: "someText", location: [1, 2, 3], style: "italics" };
    //drawText(item);
    const item2 = { text: "someText", location: [1, 2, 3], bold: true };
    drawText(item2);
        
    print("done.");
}