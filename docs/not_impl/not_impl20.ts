interface ClockConstructor {
  new (hour: number, minute: number): ClockInterface;
}
 
interface ClockInterface {
  tick(): void;
  (value: string, index: number): boolean;
}
 
function main()
{ 
	const ClockInst: ClockConstructor = class Clock implements ClockInterface {
	  constructor(h: number, m: number) {}
	  tick() {
	    print("beep beep");
	  }

	  (value: string, index: number) => true;
	};

	let clock = new ClockInst(12, 17);
	clock.tick();
	const b = clock("asd", 2);
}