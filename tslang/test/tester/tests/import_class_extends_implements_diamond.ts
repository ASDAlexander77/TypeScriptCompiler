import './export_class_extends_implements_diamond'

class Combined extends M.Base implements M.Left, M.Right {
    left: number = 2;
    right: number = 3;

    addLeft(n: number): void {
        this.left = this.left + n;
    }

    addRight(n: number): void {
        this.right = this.right + n;
    }
}

function main() {
    const c = new Combined();

    c.addBase(10);
    c.addLeft(20);
    c.addRight(30);

    assert(c.base == 11);
    assert(c.left == 22);
    assert(c.right == 33);

    // cast to each cross-module ancestor/interface type and confirm the
    // field/method resolves through the correct (non-corrupted) slot
    const asBase: M.Base = c;
    assert(asBase.base == 11);

    const asLeft: M.Left = c;
    asLeft.addLeft(100);
    assert(c.left == 122);

    const asRight: M.Right = c;
    asRight.addRight(100);
    assert(c.right == 133);

    print("done.");
}
