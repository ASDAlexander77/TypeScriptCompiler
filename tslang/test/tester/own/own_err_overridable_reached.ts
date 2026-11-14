// TSLANG_ISSUES 60: a call of a function defined here that may destroy anything only because a call it
// reaches is of a method another module may override: the error at a borrow it ends names that method
// and the way to it.
export class World {
    count = 0;
    setTransform(node: int) {
        this.count += node;
    }
}

function setNodeTransform(world: World, node: int) {
    world.setTransform(node);
}

class Prop {
    world: World;
    node: int = 1;
    constructor(world: World) {
        this.world = world;
    }

    tick() {
        setNodeTransform(this.world, this.node);
    }
}

export class Game {
    props: Prop[] = [];
    name = "game";

    tick(): int {
        const name = this.name;
        for (let i = 0; i < this.props.length; i++) {
            this.props[i].tick();
        }

        return name.length;
    }
}

function main() { print(new Game().tick()); }
