function toHex(this: number) {
}

function numberToString(n: ThisParameterType<typeof toHex>) {
}

interface Todo1 {
    title: string;
    description: string;
    completed: boolean;
}

type TodoPreview1 = Pick<Todo1, "title" | "completed">;

interface Todo2 {
    title: string;
    description: string;
    completed: boolean;
    createdAt: number;
}

type TodoPreview2 = Omit<Todo2, "description">;

function main() {
    let fiveToHex: OmitThisParameter<typeof toHex>;

    const todo1: TodoPreview1 = {
        title: "Clean room",
        completed: false,
    };

    assert(todo1.title == "Clean room");
    assert(!todo1.completed);

    const todo2: TodoPreview2 = {
        title: "Clean room",
        completed: false,
        createdAt: 1615544252770,
    };

    assert(todo2.title == "Clean room");
    assert(!todo2.completed);
    assert(todo2.createdAt === <number>1615544252770);

    print("done.");
}