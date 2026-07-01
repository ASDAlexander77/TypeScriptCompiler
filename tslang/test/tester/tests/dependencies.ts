class FileSystemObject {
    isFile(): this is File {
        return this instanceof File;
    }
    isDirectory(): this is Directory {
        return this instanceof Directory;
    }
    constructor(public path: string, private networked: boolean) { }
}

class File extends FileSystemObject {
    constructor(path: string, public content: string) {
        super(path, false);
    }
}

class Directory extends FileSystemObject {
    children: FileSystemObject[];
}

function main() {

    let fso: FileSystemObject = new File("foo/bar.txt", "foo");
    if (fso.isFile()) {
        print("fso is File");
    } else if (fso.isDirectory()) {
        print("fso is Directory");
    }

    print("done.");
}