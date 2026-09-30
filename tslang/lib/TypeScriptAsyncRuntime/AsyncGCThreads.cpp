// GC_enable_threads and the pool's thread registration with the collector, kept out of
// AsyncRuntime.cpp so that only a `gc` program - the one caller of GC_enable_threads - links them.
// See AsyncGCThreads.h.
#include "../AsyncGCThreadsCommon.inc"
