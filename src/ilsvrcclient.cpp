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
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#if 0
using namespace cv;

static double psnrhma(Mat inA, Mat inB){
	Mat a, b;
	inA.convertTo(a, CV_32F);
	inb.convertTo(b, CV_32F);
	
	double sumA  = cv::sum(a);
	double sumB  = cv::sum(b);
	double delt  = (sumA-sumB)/a.total();
	Mat bm       = b+delt;
}
#endif




/**
 * Main
 */

int   main(int argc, char* argv[]){
	void* ctx       = zmq_ctx_new();
	void* sock      = zmq_socket(ctx, ZMQ_REQ);
	char  msg[8];
	int   numBytes, N, ret;
	
	zmq_connect(sock, "tcp://127.0.0.1:5555");
	
	memset(msg, 0, sizeof(msg));
	N        = sizeof(msg);
	numBytes = zmq_send(sock, msg, N, 0);
	if(numBytes<0){
		printf("Error! Couldn't send message.\n");
		return 1;
	}else{
		printf("Sent %d bytes.\n", numBytes);
	}
	
	numBytes = zmq_recv(sock, &ret, sizeof(ret), 0);
	if(numBytes<0){
		printf("Error! Didn't receive valid reply message!\n");
		return 1;
	}else{
		printf("Request returned error code %d.\n", ret);
	}
	
	zmq_close(sock);
	zmq_ctx_term(ctx);
	return 0;
}
