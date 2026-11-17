class G<T> { @weak p: T | null = null; }
function main() { const g = new G<string>(); print(g.p === null); }
