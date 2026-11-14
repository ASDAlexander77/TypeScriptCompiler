// TSLANG_ISSUES 49: a fresh local moved into a field on some ways only is fine when on the others it
// was tested empty - not here, where `n == "y"` may be false while `l` holds an object.
class Level {
    name: string = "a";
}

class App {
    level: Level | null = null;
    read(n: string): Level | null { return n == "" ? null : new Level(); }
    start(n: string) {
        const l = this.read(n);
        if (l && n == "y") {
            this.level = l;
        }
    }
}

function main() { const a = new App(); a.start("x"); print(a.level != null); }
