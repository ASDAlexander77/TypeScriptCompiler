function main() {

    // TODO: when we process add_ we do not know exact type of 'this' for object, we can create the same logic as for class & interface with named type
    let obj = {
	val: 10,
	add: () => {
		
		function add_()
		{
			this.val ++;
		}

		print(this.val);
		add_();
		print(this.val);
	}
    };

    obj.add();
    print(obj.val);

    print("done.");
}
