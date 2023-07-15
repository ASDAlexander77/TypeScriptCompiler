export enum PointTest { Invlide, Valid };
export function pointTest (x: number, y: number): PointTest {
	return PointTest.Valid;
}