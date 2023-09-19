function getPointFactory(x: number, y: number) {
    class P {
        x = 0;
        y = 0;
    }

    return new P();
}

function main() {
    const PointZero = getPointFactory(0, 0);
    const PointOne = getPointFactory(1, 1);
    print("done.");
}