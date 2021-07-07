class Animal {
    move(distanceInMeters = 0) {
        print(`Animal moved ${distanceInMeters}m.`);
    }
}

function main() {
    const dog = new Animal();
    dog.move(10);

    const dog2 = Animal();
    dog2.move(11);
}
