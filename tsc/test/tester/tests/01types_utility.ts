type Record<K extends string | number | symbol, T> = { [P in K]: T; };
type Partial<T> = { [P in keyof T]?: T[P] | undefined; }
type Required<T> = { [P in keyof T]-?: T[P]; }
type Readonly<T> = { readonly [P in keyof T]: T[P]; }
type Pick<T, K extends keyof T> = { [P in K]: T[P]; }
type Omit<T, K extends string | number | symbol> = { [P in Exclude<keyof T, K>]: T[P]; }
type Exclude<T, U> = T extends U ? never : T;
type Extract<T, U> = T extends U ? T : never;
type NonNullable<T> = T & {};
type Parameters<T extends (...args: any) => any> = T extends (...args: infer P) => any ? P : never;

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

interface TodoPick {
    title: string;
    description: string;
    completed: boolean;
}

type TodoPreviewPick = Pick<TodoPick, "title" | "completed">;


interface TodoOmit {
    title: string;
    description: string;
    completed: boolean;
    createdAt: number;
}

type TodoPreviewOmit = Omit<TodoOmit, "description">;

type T0 = Exclude<"a" | "b" | "c", "a">;
type T1 = Exclude<"a" | "b" | "c" | "d", "a" | "b">;

type T2 = Extract<"a" | "b" | "c" | "d", "a" | "f" | "d">;

type T3 = NonNullable<string | number | undefined>;

type T4 = NonNullable<string[] | null | undefined>;


// params

type T0 = Parameters<() => string>;
type T1 = Parameters<(a: string, b: number) => string>;
type T2 = Parameters<never>;
// should error
//type T3 = Parameters<string>;

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

    const todo: Readonly<Todo> = {
        title: "Delete inactive users",
    };

    print(todo.title);

    const todoPick: TodoPreviewPick = {
        title: "Clean room",
        completed: false,
    };

    const todoOmit: TodoPreviewOmit = {
        title: "Clean room",
        completed: false,
        createdAt: 1615544252770,
    };

    let a: T0 = "b"; a = "c";
    let b: T1 = "c"; b = "d";
    let c: T2 = "a"; c = "d";

    let d: T3 = 10;
    let e: T4 = [];

    // params 
	let a: T0;
	let b: T1;
	// never type should error
	//let c: T2;

    print("done.");
}