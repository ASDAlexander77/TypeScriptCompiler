// TSLANG_ISSUES 50: a value moved into a field is read as the field after the move - but a call in
// between overwrites the field, which frees the value, so its use after the call is refused.
class Layout {
    id: int = 1;
}

class Blit {
    layout: Layout | null = null;
    make(): Layout | null { return new Layout(); }
    reset() { this.layout = new Layout(); }
    init(): int {
        const layout = this.make();
        this.layout = layout;
        this.reset();
        return layout ? layout.id : 0;
    }
}

function main() { print(new Blit().init()); }
