// A module whose class has a field of a generic class specialized on it.

export class Box<T> {
    items: T[] = [];

    [index: int]: T;

    get(index: int): T {
        return this.items[index];
    }

    set(index: int, value: T) {
        this.items[index] = value;
    }

    add(v: T) {
        this.items.push(v);
    }

    get length() {
        return this.items.length;
    }
}

export class Tree {
    name: string;
    children: Box<Tree>;

    constructor(name: string) {
        this.name = name;
        this.children = new Box<Tree>();
    }
}
