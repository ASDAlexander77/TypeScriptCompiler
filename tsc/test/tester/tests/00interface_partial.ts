interface Box {
    height: number;
    width: number;
}

interface Box {
    scale: number;
}

function main() {
    let box: Box = { height: <number>5, width: <number>6, scale: <number>10 };

    assert(box.height == 5);
    assert(box.width == 6);
    assert(box.scale == 10);

    print(box.height, box.width, box.scale);

    print("done.");
}