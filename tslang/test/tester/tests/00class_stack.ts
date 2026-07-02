// TODO: add @novtable or @novirtualtable to control generation of virtual elements
class Animal {
    move(distanceInMeters = 0) {
        print(`Animal moved ${distanceInMeters}m.`);
    }
}

function main() {
    const dog = new Animal();
    dog.move(10);

    // TODO: ERROR can't create class with vtable on stack
    // const dog2 = Animal();
    // dog2.move(11);

    print("done.");
}
