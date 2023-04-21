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

interface CatInfo {
    age: number;
    breed: string;
  }
   
type CatName = "miffy" | "boris" | "mordred";

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

	const cats: Record<CatName, CatInfo> = {
        miffy: { age: 10, breed: "Persian" },
        boris: { age: 5, breed: "Maine Coon" },
        mordred: { age: 16, breed: "British Shorthair" },
      };
  
    print(cats.boris.breed);
    assert(cats.boris.breed == "Maine Coon");
  
    print("done.");
}