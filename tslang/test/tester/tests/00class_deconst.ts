class C2 {
    x: string;
    y: string;

    constructor() {
        let self = this;                // ok
        let { x, y: y1 } = this;        // ok
        ({ x, y: y1, "y": y1 } = this); // ok
    }
}

function main()
{
    print("done.");
}