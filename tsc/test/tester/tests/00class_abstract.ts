abstract class Animal {
    constructor(protected name: string) {}

    abstract makeSound(input: string): string;

    move(meters: number) {
        print(this.name + " moved " + meters + "m.");
    }
}

class Snake extends Animal {
    constructor(name: string) {
        super(name);
    }

    makeSound(input: string): string {
        return "sssss" + input;
    }

    move() {
        print("Slithering...");
        super.move(5);
    }
}

function main() {
    const snake = new Snake("snake 1");
    snake.move();
}
