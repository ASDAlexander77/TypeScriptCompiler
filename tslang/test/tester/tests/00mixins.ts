// Mixins as the TypeScript handbook writes them: a generic function taking a class (`Base: TBase`, TBase
// constrained by a construct signature) and returning a class that extends it. Each call with a class is a
// specialization of its own, so the returned class extends that class at compile time.
type Constructor = new (...args: any[]) => {};
type GConstructor<T = {}> = new (...args: any[]) => T;

class Sprite {
    name: string = "";
    x: number = 0;
    y: number = 0;

    constructor(name: string) {
        this.name = name;
    }

    setPos(x: number, y: number) {
        this.x = x;
        this.y = y;
    }

    describe(): string {
        return "sprite " + this.name;
    }
}

// no constructor of its own
class Point {
    x: number = 1;
}

// a constructor mixin
function Scale<TBase extends Constructor>(Base: TBase) {
    return class Scaling extends Base {
        _scale: number = 1;

        setScale(scale: number) {
            this._scale = scale;
        }

        get scale(): number {
            return this._scale;
        }
    };
}

function Tagged<TBase extends Constructor>(Base: TBase) {
    return class extends Base {
        tag: string = "tagged";
    };
}

// constrained mixins: what the base must have, as a field or a method (one without a return type is a void one)
type Positionable = GConstructor<{ setPos: (x: number, y: number) => void }>;
type Describable = GConstructor<{ describe(): string }>;
type HasX = GConstructor<{ x: number }>;

function Jumpable<TBase extends Positionable>(Base: TBase) {
    return class Jumpable extends Base {
        jump() {
            this.setPos(0, 20);
        }
    };
}

function Loud<TBase extends Describable>(Base: TBase) {
    return class extends Base {
        describe(): string {
            return super.describe() + "!";
        }
    };
}

function Doubled<TBase extends HasX>(Base: TBase) {
    return class extends Base {
        double() {
            return this.x * 2;
        }
    };
}

// a class declared in the mixin's body, not returned
function makeTagged<TBase extends Constructor>(Base: TBase) {
    class Local extends Base {
        local = 4;
    }

    return new Local();
}

const EightBitSprite = Scale(Sprite);
const JumpSprite = Jumpable(Sprite);
const LoudSprite = Loud(Sprite);
const TaggedPoint = Tagged(Point);
const DoubledPoint = Doubled(Point);
// mixins applied to a mixin's class
const LoudTaggedSprite = Loud(Tagged(Sprite));
const ScaledJumpSprite = Scale(Jumpable(Sprite));

function main() {
    const flappy = new EightBitSprite("Bird");
    flappy.setScale(0.8);
    assert(flappy.name == "Bird" && flappy.scale == 0.8, "constructor mixin");
    assert(flappy instanceof Sprite, "a mixin's class extends its base");

    const jumper = new JumpSprite("J");
    jumper.jump();
    assert(jumper.y == 20, "constrained mixin calls the base's method");

    const loud = new LoudSprite("L");
    assert(loud.describe() == "sprite L!", "mixin overrides through super");
    const asSprite: Sprite = loud;
    assert(asSprite.describe() == "sprite L!", "override seen through the base");

    const point = new TaggedPoint();
    assert(point.x == 1 && point.tag == "tagged", "base without a constructor");
    assert(new DoubledPoint().double() == 2, "field constraint");

    const loudTagged = new LoudTaggedSprite("T");
    assert(loudTagged.tag == "tagged" && loudTagged.describe() == "sprite T!", "chained mixins");

    const scaledJumper = new ScaledJumpSprite("S");
    scaledJumper.jump();
    scaledJumper.setScale(2);
    assert(scaledJumper.y == 20 && scaledJumper.scale == 2, "chained constrained mixin");

    assert(makeTagged(Point).local == 4 && makeTagged(Point).x == 1, "class declared in a generic function");

    print("done.");
}
