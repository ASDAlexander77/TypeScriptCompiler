type CreateMutable<Type> = {
    -readonly [Property in keyof Type]: Type[Property];
};

type LockedAccount = {
    readonly id: string;
    readonly name: string;
};

type UnlockedAccount = CreateMutable<LockedAccount>;

function main() {
    let a: LockedAccount = { id: "id1", name: "name1" };
    let b: UnlockedAccount = { id: "id1", name: "name1" };

    assert(a.id == b.id);
    assert(a.name == b.name);

    print("done.");
}