function main() {

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
