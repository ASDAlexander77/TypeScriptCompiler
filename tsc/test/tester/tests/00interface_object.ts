let glb1 = 0;

interface Something {
    r: number;
    g: number;
    b: number;
    toString: () => string;
}

function main() {
    const something = {
        r: 11.0, g: 12.0, b: 13.0, toString() {
            if (this.b == 13.0) glb1 = 1;
            return "Hello " + this.b;
        }
    };

    const iface = <Something>something;
    print(iface.toString());

    assert(glb1 == 1);

    print("done.");
}
