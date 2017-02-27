/**
 * Benchmark ILSVRC client
 * 
 * Streams images as fast as possible from a given server.
 */

/* Includes */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>




/**
 * Main
 */

int   main(int argc, char* argv[]){
	void* ctx       = zmq_ctx_new();
	void* sock      = zmq_socket(ctx, ZMQ_REQ);
	char* msg;
	int   N;
	
	zmq_connect(sock, "tcp://127.0.0.1:5555");
	
	msg = (char*)"Hello World!";
	N   = strlen(msg);
	int   numBytes  = zmq_send(sock, msg, N+1, 0);
	if(numBytes<0){
		printf("Error! Couldn't send message.\n");
		return 1;
	}else{
		printf("Sent %d bytes.\n", numBytes);
	}
	
	zmq_close(sock);
	zmq_ctx_term(ctx);
	return 0;
}
