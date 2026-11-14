// TSLANG_ISSUES 49: a caller through `Base` takes `read`'s result for its own, since `Base.read` does not
// say it borrows; so an override may not say it does.
class Level {
    name: string = "a";
}

export class Base {
    keep: Level = new Level();
    read(): Level { return new Level(); }
}

export class Der extends Base {
    @lends("this") read(): Level { return this.keep; }
}

function main() { print(new Der().read().name); }
