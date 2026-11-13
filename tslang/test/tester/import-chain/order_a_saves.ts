// One more link between the program and the Profiler; its module sorts first. See order_program.ts.
import { State } from './order_m_state'

export class Saves extends State {
    saved = 0;
}
