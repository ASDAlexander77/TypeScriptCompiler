class Action {
    id: number;
}

class ActionA extends Action {
    value: string;
}

class ActionB extends Action {
    trueNess: boolean;
}

function main() {

/*
    const x1: Action[] = [
        { id: 2, trueness: false },
        { id: 3, name: "three" }
    ]
*/

    const x2: Action[] = [
        new ActionA(),
        new ActionB()
    ]

    const x3: Action[] = [
        new Action(),
        new ActionA(),
        new ActionB()
    ]

    const z1: { id: number }[] =
        [
            { id: 2, trueness: false },
            { id: 3, name: "three" }
        ]

    const z2: { id: number }[] =
        [
            new ActionA(),
            new ActionB()
        ]

    const z3: { id: number }[] =
        [
            new Action(),
            new ActionA(),
            new ActionB()
        ]

    print("done.");
}
