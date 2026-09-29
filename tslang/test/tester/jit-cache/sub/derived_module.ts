import '../base_module'

export class Derived extends Base {
    hello() {
        return "derived:" + super.hello();
    }
}

export function bumpFromDerived() {
    return bump();
}

// base_module's initializer has run already
export let doubled = counter * 2;
