type CreateMutable<Type> = {
    -readonly [Property in keyof Type]: Type[Property];
};

type LockedAccount = {
    readonly id: string;
    readonly name: string;
};

type UnlockedAccount = CreateMutable<LockedAccount>;

type Getters<Type> = {
    [Property in keyof Type as `get${string & Property}`]: () => Type[Property]
};

interface Person {
    name: string;
    age: number;
    location: string;
}

type LazyPerson = Getters<Person>;

type RemoveKindField<Type> = {
    [Property in keyof Type as Exclude<Property, "kind">]: Type[Property]
};

interface Circle {
    kind: "circle";
    radius: number;
}

type KindlessCircle = RemoveKindField<Circle>;

function main() {
    let a: LockedAccount = { id: "id1", name: "name1" };
    let b: UnlockedAccount = { id: "id1", name: "name1" };
    let c: LazyPerson;
    let d: KindlessCircle;

    assert(a.id == b.id);
    assert(a.name == b.name);

    print("done.");
}