// TSLANG_ISSUES 60 (-mm=own), from native_ts_graphics' core/rtxpt_game.ts: `this.props[i].tick(t)`, where
// `Prop.tick` calls `setNodeTransform`, which calls a method of the exported `GameWorld`. Another module
// may override that method, so the call may destroy anything, and the error said so only for a direct call
// of it (own_err_overridable_reached). `@final` on the class says no module does: the borrows of `this.props[i]`,
// `this.world` and `this.name` then hold over the call.

@final
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
    node: int;
    constructor(node: int) {
        this.node = node;
    }

    tick(world: World) {
        setNodeTransform(world, this.node);
    }
}

export class Game {
    world = new World();
    props: Prop[] = [];
    name = "game";

    tick(): int {
        const name = this.name;
        for (let i = 0; i < this.props.length; i++) {
            this.props[i].tick(this.world);
        }

        return name.length + this.world.count;
    }
}

function main() {
    const game = new Game();
    game.props.push(new Prop(2));
    game.props.push(new Prop(3));
    assert(game.tick() == 9);
    assert(game.tick() == 14);
    print("done.");
}
