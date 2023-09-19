let glb1 = 0;

interface Something {
    r: number;
    g: number;
    b: number;
    toString: () => string;
}

function testIFace(iface: Something) {
    print(iface.toString());
}

interface Surface {
    n: number;
}

// global const object
let shiny: Surface = {
    n: 10.0
}

// to cause creating VTABLE earlier
function create_vtable_to_repeat_bug() {
    const something = {
        r: 11.0, g: 12.0, b: 13.0, toString() {
            if (this.b == 13.0) glb1++;
            return "Hello " + this.b;
        }
    };

    const iface = <Something>something;
}

function main() {
    const something = {
        r: 11.0, g: 12.0, b: 13.0, toString() {
            if (this.b == 13.0) glb1++;
            return "Hello " + this.b;
        }
    };

    const iface = <Something>something;
    print(iface.toString());

    testIFace(something);

    assert(glb1 == 2);

    assert(shiny.n == 10.0);

    print("done.");
}
