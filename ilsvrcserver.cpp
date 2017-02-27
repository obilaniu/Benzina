/**
 * ILSVRC server.
 * 
 *   Fetches, decodes and serves up dataset images as fast as possible.
 * 
 * USAGE:
 * 
 * ilsvrcserver [args] path/to/dataset.hdf5 <port>
 * 
 *   -r       madvise() for random access.
 *   -s       madvise() for sequential access.
 *   -l       mlock()   the dataset.
 *   <port>   Localhost port number to bind to. Default 5555.
 * 
 * IMPLEMENTATION:
 * 
 *   The server is stateless, and accepts only a single ZeroMQ request packet,
 *   whose payload is as follows:
 *
 * PACKET FORMAT:
 *   Version:      00 00 00 00
 *   Request#:     xx xx xx xx
 *   {...}
 * 
 * Request#: 00 00 00 00 (Sequential Batch)
 *   Batch Size:   xx xx xx xx xx xx xx xx
 *   Start Index:  xx xx xx xx xx xx xx xx
 *   X Off:        xx xx xx xx xx xx xx xx
 *   Y Off:        xx xx xx xx xx xx xx xx
 *   X Path Len:   xx xx xx xx xx xx xx xx
 *   Y Path Len:   xx xx xx xx xx xx xx xx
 *   X Path:       {xx}*X Path Len 00
 *   Y Path:       {xx}*Y Path Len 00
 * 
 * Request#: 01 00 00 00 (Random Batch)
 *   Batch Size:   xx xx xx xx xx xx xx xx
 *   Indexes:      {xx xx xx xx xx xx xx xx}*Batch Size
 *   X Off:        xx xx xx xx xx xx xx xx
 *   Y Off:        xx xx xx xx xx xx xx xx
 *   X Path Len:   xx xx xx xx xx xx xx xx
 *   Y Path Len:   xx xx xx xx xx xx xx xx
 *   X Path:       {xx}*X Path Len 00
 *   Y Path:       {xx}*Y Path Len 00
 * 
 */

/* Includes */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>


/* Data Structure Definitions */

/**
 * Server State
 */

struct ILSVRC;
typedef struct ILSVRC ILSVRC;
struct ILSVRC{
	int       bindPort;
	char*     pathDataset;
	int       madvMode;   /* 0: Normal  1: Random  2: Sequential */
	int       mlockMode;  /* 0: Off     1: On */
	int       exit;
	int       exitCode;
	char      zmqBindAddr[256];
	void*     zmqCtx;
	void*     zmqSock;
};

struct ILSVRC_CONN;
typedef struct ILSVRC_CONN ILSVRC_CONN;
struct ILSVRC_CONN{
	zmq_msg_t zmqMsgId;
	zmq_msg_t zmqMsgData;
	zmq_msg_t zmqMsgDis;
};


/* Static Function Prototypes */
static int   ilsvrcHasRequestedExit   (ILSVRC* s);
static void  ilsvrcRequestExitWithCode(ILSVRC* s, int ret);
static int   ilsvrcParseArgs          (ILSVRC* s, int argc, char** argv);
static int   ilsvrcPrintHelp          (ILSVRC* s);
static int   ilsvrcZmqSetup           (ILSVRC* s);
static int   ilsvrcEventLoop          (ILSVRC* s);
static void  ilsvrcHandleEvent        (ILSVRC* s);
static int   ilsvrcTeardown           (ILSVRC* s, int ret);



/**
 * Has exit been requested?
 */

static int   ilsvrcHasRequestedExit(ILSVRC* s){
	return !!s->exit;
}

/**
 * Has exit been requested?
 */

static void  ilsvrcRequestExitWithCode(ILSVRC* s, int ret){
	s->exit     = 1;
	s->exitCode = ret;
}

/**
 * Parse arguments.
 */

static int   ilsvrcParseArgs(ILSVRC* s, int argc, char** argv){
	int i;
	char* p;
	
	memset(s, 0, sizeof(*s));
	s->bindPort = 5555;
	
	if(argc <= 1){
		return ilsvrcPrintHelp(s);
	}
	
	for(i=1;i<argc;i++){
		if(!s->pathDataset){
			/**
			 * We haven't seen the path to the dataset yet, this should be either a
			 * switch or the path.
			 */
			
			if(*argv[i] == '-'){
				/* It's a switch */
				p = argv[i]+1;
				
				while(*p){
					switch(*p){
						case 'l': s->mlockMode = 1; break;
						case 'n': s->madvMode  = 0; break;
						case 'r': s->madvMode  = 1; break;
						case 's': s->madvMode  = 2; break;
						default: return ilsvrcPrintHelp(s);
					}
					p++;
				}
			}else{
				/* It's the path to the dataset. */
				s->pathDataset = argv[i];
			}
		}else{
			/* This must be the port number */
			s->bindPort = strtoul(argv[i], &p, 0);
			if(*argv[i] == '\0' || *p == '\0'){
				return ilsvrcPrintHelp(s);
			}
		}
	}
	
	return ilsvrcZmqSetup(s);
}

/**
 * Print Help and quit
 */

static int   ilsvrcPrintHelp(ILSVRC* s){
	fprintf(stderr, "Usage: ilsvrcserver [-rsnl] path/to/dataset.hdf5 <port>\n");
	
	return ilsvrcTeardown(s, 1);
}

/**
 * Setup Zmq
 */

static int   ilsvrcZmqSetup(ILSVRC* s){
	/* Create the various ZMQ objects */
	s->zmqCtx  = zmq_ctx_new();
	s->zmqSock = zmq_socket(s->zmqCtx, ZMQ_ROUTER);
	sprintf(s->zmqBindAddr, "tcp://127.0.0.1:%d", s->bindPort);
	zmq_bind(s->zmqSock, s->zmqBindAddr);
	
	/* Print summary: */
	fprintf(stdout, "ILSVRC Server started.\n");
	fprintf(stdout, "Port#:     %d\n",   s->bindPort);
	fprintf(stdout, "Dataset:   '%s'\n", s->pathDataset);
	fprintf(stdout, "Transport: '%s'\n", s->zmqBindAddr);
	fprintf(stdout, "\n");
	fflush (stdout);
	
	/* Enter Event Loop. */
	return ilsvrcEventLoop(s);
}

/**
 * Handle one event loop iteration
 */

static int   ilsvrcEventLoop(ILSVRC* s){
	while(!ilsvrcHasRequestedExit(s)){
		ilsvrcHandleEvent(s);
	}
	
	return ilsvrcTeardown(s, 0);
}

/**
 * Handle one event.
 */

static void  ilsvrcHandleEvent     (ILSVRC* s){
	ILSVRC_CONN c_STACK, *c = &c_STACK;
	int         hasMore = 1, nbId, nbMsg, nbDis;
	size_t      optlen = sizeof(hasMore);
	zmq_msg_init(&c->zmqMsgId);
	zmq_msg_init(&c->zmqMsgData);
	zmq_msg_init(&c->zmqMsgDis);
	
	/**
	 * A valid ZMQ message will be in three parts:
	 *   - Identity
	 *   - Empty Delimiter
	 *   - Payload
	 */
	
	nbDis = zmq_msg_recv(&c->zmqMsgDis, s->zmqSock, 0);
	if(nbDis  <= 0){
		printf("Error (ID receive)! %s\n", zmq_strerror(zmq_errno()));
		goto discard;
	}else{
		zmq_msg_move(&c->zmqMsgId,   &c->zmqMsgDis);
		nbId   = nbDis;
		(void)nbId;
	}
	nbDis = zmq_msg_recv(&c->zmqMsgDis, s->zmqSock, 0);
	if(nbDis != 0){
		printf("Error (delimiter receive)! %s\n", zmq_strerror(zmq_errno()));
		goto discard;
	}
	nbDis = zmq_msg_recv(&c->zmqMsgDis, s->zmqSock, 0);
	if(nbDis<0){
		printf("Error (payload receive)! %s\n", zmq_strerror(zmq_errno()));
		goto discard;
	}else{
		zmq_msg_move(&c->zmqMsgData, &c->zmqMsgDis);
		nbMsg  = nbDis;
		printf("%d of payload.\n", nbMsg);
	}
	printf("'%s'\n", (const char*)zmq_msg_data(&c->zmqMsgData));
	fflush(stdout);
	
	/* Discard. */
	discard:
	while(zmq_getsockopt(s->zmqSock, ZMQ_RCVMORE, &hasMore, &optlen) == 0 && hasMore){
		nbDis = zmq_msg_recv(&c->zmqMsgDis, s->zmqSock, 0);
	}
	
	
	zmq_msg_close(&c->zmqMsgId);
	zmq_msg_close(&c->zmqMsgData);
	zmq_msg_close(&c->zmqMsgDis);
}

/**
 * Tear down the server.
 */

static int   ilsvrcTeardown(ILSVRC* s, int ret){
	ilsvrcRequestExitWithCode(s, ret);
	
	zmq_close(s->zmqSock);
	zmq_ctx_term(s->zmqCtx);
	
	return s->exitCode;
}


/**
 * Main
 */

int   main(int argc, char* argv[]){
	ILSVRC s_STACK, *s = &s_STACK;
	
	return ilsvrcParseArgs(s, argc, argv);
}
