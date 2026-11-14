// TSLANG_ISSUES 49: one field moved into another and the source cleared - but a call in between may
// free or read what both fields hold, so the move is refused.
class Level {
    name: string = "a";
}

class App {
    level: Level | null = new Level();
    other: Level | null = null;
    side() { this.other = null; }
    moveAside() {
        this.other = this.level;
        this.side();
        this.level = null;
    }
}

function main() { const a = new App(); a.moveAside(); print(a.other != null); }
