type Partial<T> = { [P in keyof T]?: T[P] | undefined; }

interface Todo {
  title: string;
  description: string;
}
 
function updateTodo(todo: Todo, fieldsToUpdate: Partial<Todo>) {
  //return { ...todo, ...fieldsToUpdate };
  return { ...todo };
}

function main()
{
	const todo1 = {
	  title: "organize desk",
	  description: "clear clutter",
	};
 
	const todo2 = updateTodo(todo1, {
	  description: "throw out trash",
	});

	print("done.");

}