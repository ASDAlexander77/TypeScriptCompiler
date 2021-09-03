#ifndef CONFIG_H_
#define CONFIG_H_

//#define GC_ENABLE 1
//#define TSGC_ENABLE 1

#define ENABLE_ASYNC 1

#define USE_SPRINTF 1
#ifndef WIN32
#ifndef USE_SPRINTF
#define USE_SPRINTF 1
#endif
#endif

#endif // CONFIG_H_