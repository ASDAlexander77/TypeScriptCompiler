type BirdType = {
    wings: 2;
};

interface BirdInterface {
    wings: 2;
}

const bird1: BirdType = { wings: 2 };
const bird2: BirdInterface = { wings: 2 };

// Because TypeScript is a structural type system,
// it's possible to intermix their use too.

const bird3: BirdInterface = bird1;

// They both support extending other interfaces and types.
// Type aliases do this via intersection types, while
// interfaces have a keyword.

type Owl = { nocturnal: true } & BirdType;
type Robin = { nocturnal: false } & BirdInterface;

interface Peacock extends BirdType {
    colourful: true;
    flies: false;
}

interface Chicken extends BirdInterface {
    colourful: false;
    flies: false;
}

let owl: Owl = { wings: 2, nocturnal: true };
let chicken: Chicken = { wings: 2, colourful: false, flies: false };

function main() {
    print("done.");
}