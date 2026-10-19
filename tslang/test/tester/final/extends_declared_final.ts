// The default library's shape: a @final declaration (lib.d.ts) cannot be extended either.
@final
declare class D {
    m(): int;
}

class E extends D {}

function main() {
}
