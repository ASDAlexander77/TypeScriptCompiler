namespace M {

    // Base class exported so another module can `extends` it - covers the
    // cross-module analogue of 00class_super.ts's same-module class extends,
    // which was never verified across the module boundary (unlike interface
    // extends, which got dedicated cross-module coverage in #268-#270).

    export class Animal {
        name: string;

        constructor(name: string) {
            this.name = name;
        }

        speak(): string {
            return `${this.name} makes a noise.`;
        }
    }
}
