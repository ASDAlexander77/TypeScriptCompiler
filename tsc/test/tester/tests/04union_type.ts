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

    const animal = { type: 'cat', canMeow: true } as Animal;

    print("done.");
}
