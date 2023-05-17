// @strict: true
interface Cat {
    type: 'cat';
    canMeow: true;
}

interface Dog {
    type: 'dog';
    canBark: true;
}

type Animal = Cat | Dog;

function main() {

    let test = false;
    const animal = { type: 'dog', canBark: true } as Animal;

    if (animal.type == "cat") {
        print("this is cat, can meow? ", animal.canMeow);
    }

    if (animal.type == "dog") {
        test = true;
        print("this is dog, can bark? ", animal.canBark);
    }

    assert(test);

    print("done.");
}
