// A function another module may call that keeps a parameter for good has to say so: its callers in
// other modules make the argument immortal only if they know.
type Callback = (n: number) => void;
let retained: Callback[] = [];

export class Registry {
    register(handler: Callback): void {
        retained.push(handler);
    }
}
