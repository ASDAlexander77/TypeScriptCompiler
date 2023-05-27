// @target: es5
// @declaration: true
// There's a 'File' class in the stdlib, wrap with a namespace to avoid collision
class FileSystemObject {
    isFSO: this is FileSystemObject;
    get isFile(): this is File {
        return this instanceof File;
    }
    set isFile(param) {
        // noop
    }
    get isDirectory(): this is Directory {
        return this instanceof Directory;
    }
    isNetworked: this is (Networked & this);
    constructor(public path: string) { }
}

class File extends FileSystemObject {
    constructor(path: string, public content: string) { super(path); }
}

class Directory extends FileSystemObject {
    children: FileSystemObject[];
}

interface Networked {
    host: string;
}

interface GenericLeadGuard<T> extends GenericGuard<T> {
    lead(): void;
}

interface GenericFollowerGuard<T> extends GenericGuard<T> {
    follow(): void;
}

interface GenericGuard<T> {
    target: T;
    isLeader: this is (GenericLeadGuard<T>);
    isFollower: this is GenericFollowerGuard<T>;
}

interface SpecificGuard {
    isMoreSpecific: this is MoreSpecificGuard;
}

interface MoreSpecificGuard extends SpecificGuard {
    do(): void;
}

function main() {
    let file: FileSystemObject = new File("foo/bar.txt", "foo");
    file.isNetworked = false;
    file.isFSO = file.isFile;
    //file.isFile = true;
    let x = file.isFile;
    if (file.isFile) {
        file.content;
        if (file.isNetworked) {
	    // TODO: finish it
            //file.host;
            //file.content;
        }
    }
    else if (file.isDirectory) {
        file.children;
    }
    else if (file.isNetworked) {
        // TODO: finish it
        //file.host;
    }

    let guard: GenericGuard<File> = {};
    if (guard.isLeader) {
        guard.lead();
    }
    else if (guard.isFollower) {
        guard.follow();
    }

    let general: SpecificGuard = {};
    if (general.isMoreSpecific) {
        general.do();
    }

    print("done.");
}
