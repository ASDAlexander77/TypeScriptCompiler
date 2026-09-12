namespace M {

    // Exports every shape whose returned reference an importing module has to take over:
    // a plain method, a virtual one that a subclass overrides, and one reached through an
    // interface. All of them hand back a freshly built string, so a caller that keeps
    // retaining leaks and a caller that releases twice frees a live one.
    // See docs/reference-counting-evaluation.md item 5al.

    export interface Describable {
        describe(): string;
    }

    export class Animal implements Describable {
        name: string;

        constructor(name: string) {
            this.name = name;
        }

        speak(): string {
            return `${this.name} makes a noise.`;
        }

        describe(): string {
            return `animal ${this.name}`;
        }
    }

    export function greet(who: string): string {
        return `hello ${who}`;
    }
}
