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

class Color3 {
    constructor(public r: number,
        public g: number,
        public b: number) {
    }

    static white = 1 + 1;
}


function main() {
    assert(glb1 == 2);
    assert(Color3.white == 2);
    print("done.");
}
