class B {
    constructor(x: number, y: number, ...z: string[]) { print("constr", x, y); }
}

function main() {

    let dd : {
        new(x: number, y: number, ...z: string[]);
    } = B;

    const ss = new dd(1, 2);

    print("done");
}