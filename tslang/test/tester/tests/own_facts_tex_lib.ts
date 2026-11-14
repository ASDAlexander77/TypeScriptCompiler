// TSLANG 74's library (own-facts-objects.cmake): a texture loader whose result is the caller's own,
// or null.
@final
export class Texture {
    constructor(public size: number) {}

    format(): number {
        return 4;
    }

    view(format: number): number {
        return this.size * format;
    }
}

@final
export class Device {
    load(name: string): Texture | null {
        return name.length > 1 ? new Texture(name.length) : null;
    }
}
