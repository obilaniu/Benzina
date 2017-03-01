/**
 * ILSVRC server.
 * 
 *   Fetches, decodes and serves up dataset images as fast as possible.
 * 
 * USAGE:
 * 
 * ilsvrcserver [args] path/to/dataset.hdf5
 * 
 *   -d       madvise() for dont-need.
 *   -n       madvise() for normal access.
 *   -r       madvise() for random access.
 *   -s       madvise() for sequential access.
 *   -l       mlock()   the dataset into memory.
 *   -p       Localhost port number to bind to. Default 5555.
 * 
 * IMPLEMENTATION:
 * 
 *   The server is stateless, and accepts three ZeroMQ request packets,
 *   whose payload are as follows:
 *
 * PACKET FORMAT:
 *   Version:      00 00 00 00
 *   Request#:     xx xx xx xx
 *   {...}
 * 
 * Request#: 00 00 00 00 (Exit)
 * 
 * Request#: 01 00 00 00 (Sequential Batch)
 *   Batch Size:   xx xx xx xx xx xx xx xx
 *   Start Index:  xx xx xx xx xx xx xx xx
 *   X Off:        xx xx xx xx xx xx xx xx
 *   Y Off:        xx xx xx xx xx xx xx xx
 *   X Path Len:   xx xx xx xx xx xx xx xx
 *   Y Path Len:   xx xx xx xx xx xx xx xx
 *   X Path:       {xx}*X Path Len 00
 *   Y Path:       {xx}*Y Path Len 00
 * 
 * Request#: 02 00 00 00 (Random Batch)
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
	int       dontneed;   /* 0: Off     1: On */
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
	zmq_msg_t   zmqMsgId;
	zmq_msg_t   zmqMsgDel;
	zmq_msg_t   zmqMsgData;
	zmq_msg_t   zmqMsgDis;
	const char* msg;
	size_t      msgLen;
	uint32_t    ver;
	uint32_t    req;
};


/* Static Function Prototypes */
static int   ilsvrcHasRequestedExit          (ILSVRC* s);
static void  ilsvrcRequestExitWithCode       (ILSVRC* s, int ret);
static int   ilsvrcParseArgs                 (ILSVRC* s, int argc, char** argv);
static int   ilsvrcPrintHelp                 (ILSVRC* s);
static int   ilsvrcZmqSetup                  (ILSVRC* s);
static int   ilsvrcEventLoop                 (ILSVRC* s);
static int   ilsvrcHandleEvent               (ILSVRC* s);
static int   ilsvrcTeardown                  (ILSVRC* s, int ret);
static int   ilsvrcConnInit                  (ILSVRC* s, ILSVRC_CONN* c);
static int   ilsvrcConnAccept                (ILSVRC* s, ILSVRC_CONN* c);
static int   ilsvrcConnProcess               (ILSVRC* s, ILSVRC_CONN* c);
static int   ilsvrcConnReply                 (ILSVRC* s, ILSVRC_CONN* c, int ret);
static int   ilsvrcConnDiscard               (ILSVRC* s, ILSVRC_CONN* c);
static int   ilsvrcConnCleanup               (ILSVRC* s, ILSVRC_CONN* c);
static int   ilsvrcConnRequestExit           (ILSVRC* s, ILSVRC_CONN* c);
static int   ilsvrcConnRequestSequentialBatch(ILSVRC* s, ILSVRC_CONN* c);
static int   ilsvrcConnRequestRandomBatch    (ILSVRC* s, ILSVRC_CONN* c);



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
						case 'd': s->dontneed  = 1; break;
						case 'l': s->mlockMode = 1; break;
						case 'n': s->madvMode  = 0; break;
						case 'r': s->madvMode  = 1; break;
						case 's': s->madvMode  = 2; break;
						case 'p':
							/** 
							 * -p cannot be followed by another single-letter option
							 * and must be followed by a parseable port number.
							 */
							
							if(p[1] != '\0' || argv[++i] == NULL){
								return ilsvrcPrintHelp(s);
							}
							
							/* This must be the port number */
							s->bindPort = strtoul(argv[i], &p, 0);
							if(*argv[i] == '\0' || *p != '\0'){
								return ilsvrcPrintHelp(s);
							}
						break;
						default: return ilsvrcPrintHelp(s);
					}
					p++;
				}
			}else{
				/* It's the path to the dataset. */
				s->pathDataset = argv[i];
			}
		}
	}
	
	return ilsvrcZmqSetup(s);
}

/**
 * Print Help and quit
 */

static int   ilsvrcPrintHelp(ILSVRC* s){
	fprintf(stderr, "Usage: ilsvrcserver [-drsnl -p <port>] path/to/dataset.hdf5\n");
	
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

static int   ilsvrcHandleEvent     (ILSVRC* s){
	ILSVRC_CONN c_STACK, *c = &c_STACK;
	
	return ilsvrcConnInit(s, c);
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
 * Initialize connection.
 */

static int   ilsvrcConnInit(ILSVRC* s, ILSVRC_CONN* c){
	memset(c, 0, sizeof(*c));
	zmq_msg_init(&c->zmqMsgId);
	zmq_msg_init(&c->zmqMsgDel);
	zmq_msg_init(&c->zmqMsgData);
	zmq_msg_init(&c->zmqMsgDis);
	
	return ilsvrcConnAccept(s, c);
}

/**
 * Accept a connection.
 */

static int   ilsvrcConnAccept(ILSVRC* s, ILSVRC_CONN* c){
	/**
	 * A valid ZMQ message will be in three parts:
	 *   - Identity
	 *   - Empty Delimiter
	 *   - Payload
	 */
	
	if(zmq_msg_recv(&c->zmqMsgDis, s->zmqSock, 0)  <= 0){
		fprintf(stderr, "Error (ID receive)! %s\n",        zmq_strerror(zmq_errno()));
		return ilsvrcConnDiscard(s, c);
	}
	zmq_msg_move(&c->zmqMsgId,   &c->zmqMsgDis);
	if(zmq_msg_recv(&c->zmqMsgDis, s->zmqSock, 0)  != 0){
		fprintf(stderr, "Error (delimiter receive)! %s\n", zmq_strerror(zmq_errno()));
		return ilsvrcConnDiscard(s, c);
	}
	zmq_msg_move(&c->zmqMsgDel,  &c->zmqMsgDis);
	if(zmq_msg_recv(&c->zmqMsgDis, s->zmqSock, 0)   < 0){
		fprintf(stderr, "Error (payload receive)! %s\n",   zmq_strerror(zmq_errno()));
		return ilsvrcConnDiscard(s, c);
	}
	zmq_msg_move(&c->zmqMsgData, &c->zmqMsgDis);
	c->msg    = (const char*)zmq_msg_data(&c->zmqMsgData);
	c->msgLen = (size_t)     zmq_msg_size(&c->zmqMsgData);
	
	return ilsvrcConnProcess(s, c);
}

/**
 * Process request.
 */

static int   ilsvrcConnProcess(ILSVRC* s, ILSVRC_CONN* c){
	if(c->msgLen < 8){
		return ilsvrcConnReply(s, c, 1);
	}
	
	memcpy(&c->ver, c->msg+0, sizeof(c->ver));
	memcpy(&c->req, c->msg+4, sizeof(c->req));
	
	if(c->ver > 0){
		return ilsvrcConnReply(s, c, 2);
	}
	
	switch(c->req){
		case 0:  return ilsvrcConnRequestExit           (s, c);
		case 1:  return ilsvrcConnRequestSequentialBatch(s, c);
		case 2:  return ilsvrcConnRequestRandomBatch    (s, c);
		default: return ilsvrcConnReply                 (s, c, 3);
	}
}

/**
 * Send reply packet back. This packet will contain 4 bytes only, a return
 * code of sorts.
 */

static int   ilsvrcConnReply(ILSVRC* s, ILSVRC_CONN* c, int ret){
	/**
	 * A valid ZMQ message will be in three parts:
	 *   - Identity
	 *   - Empty Delimiter
	 *   - Payload
	 */
	
	if(zmq_msg_send(&c->zmqMsgId,   s->zmqSock, ZMQ_SNDMORE)  <= 0){
		fprintf(stderr, "Error (ID send)! %s\n",        zmq_strerror(zmq_errno()));
		return ilsvrcConnDiscard(s, c);
	}
	if(zmq_msg_send(&c->zmqMsgDel,  s->zmqSock, ZMQ_SNDMORE)  != 0){
		fprintf(stderr, "Error (delimiter send)! %s\n", zmq_strerror(zmq_errno()));
		return ilsvrcConnDiscard(s, c);
	}
	if(zmq_send    (s->zmqSock, &ret, sizeof(ret),        0)  != sizeof(ret)){
		fprintf(stderr, "Error (payload send)! %s\n",   zmq_strerror(zmq_errno()));
		return ilsvrcConnDiscard(s, c);
	}
	
	return ilsvrcConnDiscard(s, c);
}

/**
 * Discard remaining packets in the message.
 */

static int   ilsvrcConnDiscard(ILSVRC* s, ILSVRC_CONN* c){
	int    hasMore = 0;
	size_t optlen  = sizeof(hasMore);
	
	while(zmq_getsockopt(s->zmqSock, ZMQ_RCVMORE, &hasMore, &optlen) == 0 &&
	      hasMore                                                         &&
	      zmq_msg_recv(&c->zmqMsgDis, s->zmqSock, 0)                 >= 0){}
	
	return ilsvrcConnCleanup(s, c);
}

/**
 * Clean up connection.
 */

static int   ilsvrcConnCleanup(ILSVRC* s, ILSVRC_CONN* c){
	zmq_msg_close(&c->zmqMsgId);
	zmq_msg_close(&c->zmqMsgDel);
	zmq_msg_close(&c->zmqMsgData);
	zmq_msg_close(&c->zmqMsgDis);
	
	return 0;
}

/**
 * Message: Exit
 */

static int   ilsvrcConnRequestExit           (ILSVRC* s, ILSVRC_CONN* c){
	ilsvrcRequestExitWithCode(s, 0);
	return ilsvrcConnReply(s, c, 0);
}

/**
 * Message: Sequential Batch
 */

static int   ilsvrcConnRequestSequentialBatch(ILSVRC* s, ILSVRC_CONN* c){
	uint64_t    batchSize, startIndex, xOff, yOff, xPathLen, yPathLen;
	const char* xPath, * yPath;
	
	/**
	 * Packet too short?
	 */
	
	if(c->msgLen < 60){
		return ilsvrcConnReply(s, c, 4);
	}
	
	/**
	 * Parse header.
	 */
	
	memcpy(&batchSize,    c->msg+ 8, sizeof(batchSize));
	memcpy(&startIndex,   c->msg+16, sizeof(startIndex));
	memcpy(&xOff,         c->msg+24, sizeof(xOff));
	memcpy(&yOff,         c->msg+32, sizeof(yOff));
	memcpy(&xPathLen,     c->msg+40, sizeof(xPathLen));
	memcpy(&yPathLen,     c->msg+48, sizeof(yPathLen));
	
	/**
	 * Packet of incorrect length?
	 */
	
	if(c->msgLen != 56+xPathLen+1+yPathLen+1){
		return ilsvrcConnReply(s, c, 4);
	}
	
	/**
	 * Check sanity of paths given to us.
	 */
	
	xPath = c->msg + 56;
	yPath = c->msg + 56 + xPathLen + 1;
	if(xPath[xPathLen] != '\0'   || yPath[yPathLen] != '\0'  ||
	   strlen(xPath) != xPathLen || strlen(yPath) != yPathLen){
		return ilsvrcConnReply(s, c, 4);
	}
	
	/**
	 * The request is valid, handle it here.
	 */
	
	//.........
	
	
	return ilsvrcConnReply(s, c, 0);
}

/**
 * Message: Random Batch
 */

static int   ilsvrcConnRequestRandomBatch    (ILSVRC* s, ILSVRC_CONN* c){
	uint64_t    batchSize, * indexes, xOff, yOff, xPathLen, yPathLen;
	const char* xPath, * yPath;
	
	/**
	 * Packet too short?
	 */
	
	if(c->msgLen < 60){
		return ilsvrcConnReply(s, c, 4);
	}
	
	/**
	 * Parse header.
	 */
	
	memcpy(&batchSize,    c->msg+ 8, sizeof(batchSize));
	indexes = (uint64_t*)(c->msg+16);
	
	/**
	 * Packet of incorrect length?
	 */
	
	if(c->msgLen != 48 + batchSize*8 + xPathLen+1+yPathLen+1){
		return ilsvrcConnReply(s, c, 4);
	}
	
	memcpy(&xOff,         c->msg+16+batchSize*8, sizeof(xOff));
	memcpy(&yOff,         c->msg+24+batchSize*8, sizeof(yOff));
	memcpy(&xPathLen,     c->msg+32+batchSize*8, sizeof(xPathLen));
	memcpy(&yPathLen,     c->msg+40+batchSize*8, sizeof(yPathLen));
	
	/**
	 * Check sanity of paths given to us.
	 */
	
	xPath = c->msg + 48 + batchSize*8;
	yPath = c->msg + 48 + batchSize*8 + xPathLen + 1;
	if(xPath[xPathLen] != '\0'   || yPath[yPathLen] != '\0'  ||
	   strlen(xPath) != xPathLen || strlen(yPath) != yPathLen){
		return ilsvrcConnReply(s, c, 4);
	}
	
	/**
	 * The request is valid, handle it here.
	 */
	
	//.........
	(void)indexes;
	
	
	return ilsvrcConnReply(s, c, 0);
}

/**
 * Main
 */

int   main(int argc, char* argv[]){
	ILSVRC s_STACK, *s = &s_STACK;
	
	return ilsvrcParseArgs(s, argc, argv);
}
