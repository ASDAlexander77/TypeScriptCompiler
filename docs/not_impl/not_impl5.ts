function foo(x = class { static prop: string }): string {
    return undefined;
}

function main()
{
  foo(class { static prop = "hello" }).length;
}