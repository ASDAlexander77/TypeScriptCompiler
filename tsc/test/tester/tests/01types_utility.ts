type Record<K extends string | number | symbol, T> = { [P in K]: T; };
type Partial<T> = { [P in keyof T]?: T[P] | undefined; }
type Required<T> = { [P in keyof T]-?: T[P]; }

interface CatInfo {
    age: number;
    breed: string;
}

type CatName = "miffy" | "boris" | "mordred";

interface Todo {
    title: string;
    description: string;
}

interface Props {
    a?: number;
    b?: string;
}

function updateTodo(todo: Todo, fieldsToUpdate: Partial<Todo>) {
    return { /*...todo,*/ ...fieldsToUpdate };
}

function main() {
    const cats: Record<CatName, CatInfo> = {
        miffy: { age: 10, breed: "Persian" },
        boris: { age: 5, breed: "Maine Coon" },
        mordred: { age: 16, breed: "British Shorthair" },
    };

    print(cats.boris.breed);

    const todo1 = {
        title: "organize desk",
        description: "clear clutter",
    };

    const todo2 = updateTodo(todo1, {
        description: "throw out trash",
    });

    assert(todo2.description === "throw out trash");

    const obj: Props = { a: 5 };

    const obj2: Required<Props> = { a: 5 };

    print(obj.a, obj2.a);

    print("done.");
}