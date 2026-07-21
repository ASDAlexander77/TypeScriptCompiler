namespace M {

    // 3-level chain with the first two levels defined (and already linked to
    // each other) in the exporting module, so the importer's further-derived
    // class must extend a base whose OWN vtable was built across a module
    // boundary already - this is the class analogue of the cross-module
    // interface multilevel coverage (extends_interface_multilevel.ts), which
    // found real vtable-offset bugs when a 2nd/3rd extends target's methods
    // were mis-patched.

    export class A {
        a: number = 1;

        addA(n: number): void {
            this.a = this.a + n;
        }

        describe(): string {
            return `A:${this.a}`;
        }
    }

    export class B extends A {
        b: number = 2;

        addB(n: number): void {
            this.b = this.b + n;
        }

        describe(): string {
            return `B:${this.b}/${super.describe()}`;
        }
    }
}
