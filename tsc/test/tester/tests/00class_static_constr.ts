let glb1 = 0;

class Color {
    static constructor() {
        glb1++;
        print("Static construct");
    }

    constructor(public r: number,
        public g: number,
        public b: number) {
    }

    static white = 1;
}

class Color2 {
    static constructor() {
        glb1++;
        print("Static construct 2");
    }
}

function main() {
    assert(glb1 == 2);
    print("done.");
}
