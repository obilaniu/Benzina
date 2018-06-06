/* Includes */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "benzina/benzina.h"


/* Defines */
#define CTRL_SOCK     3



/**
 * @brief Run Fork Server.
 * 
 * The fork server expects that on FD 3 there will be one end of a UNIX-domain,
 * SOCK_SEQPACKET-type socket.
 */

int main(int argc, char* argv[]){
	const char* childargv[] = {
		"/usr/bin/ls",
		"-la",
		"/proc/self/maps",
		NULL
	};
	extern const char** environ;
	
	if(benzinaForkServerDoLaunch(argc > 1 ? argv[1] : "./benzina_fork_server") < 0){
		printf("Failed to launch benzina_fork_server!\n");
		exit(EXIT_FAILURE);
	}
	if(benzinaForkServerSpawn("/usr/bin/ls", "/", childargv, environ, 0) < 0){
		printf("Failed to execute test through forkserver!\n");
		exit(EXIT_FAILURE);
	}
	return EXIT_SUCCESS;
}
