function getPointFactory(x: number, y: number) {
    class P {
        x = x;
        y = y;
    }

    return P;
}

function main() {
    const PointZero = getPointFactory(0, 0);
    const PointOne = getPointFactory(1, 1);
}