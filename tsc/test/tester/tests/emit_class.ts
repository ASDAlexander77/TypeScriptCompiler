@dynamic
import './decl_class';

function main() {
	const a1 = new Account();
	print(a1.n);

	const a2 = new Account(2);
	print(a2.n);

    print(Account.data);

    assert(Account.data == 2);

    print (a2 instanceof Account);

    assert(a2 instanceof Account);

	print("done.");
}