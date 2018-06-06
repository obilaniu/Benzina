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
#include <sys/wait.h>
#include <unistd.h>

#include "benzinafork.h"
#include "benzina/benzina.h"


/* Defines */
#define CTRL_SOCK     3

#define EXIT_BADCLEAN 1
#define EXIT_BADALLOC 2
#define EXIT_BADMSG   3



/* Function Definitions */

/**
 * @brief Close all file descriptors.
 * 
 * As a forkserver, we must attempt to provide as pristine an environment as
 * possible for the processes that fork out of it. This includes closing all
 * file descriptors the forkserver may have inherited from its parents.
 */

void closeallfds(void){
	char* s = NULL;
	struct dirent* dent;
	int  dfd, efd;
	DIR* d;
	
	/* Get directory file descriptor manually, so we know its value. */
	dfd = open("/proc/self/fd", O_RDONLY|O_CLOEXEC|O_DIRECTORY);
	if(dfd < 1){
		exit(EXIT_BADCLEAN);
	}
	
	/* Open the directory stream from its descriptor. */
	d = fdopendir(dfd);
	if(!d){
		exit(EXIT_BADCLEAN);
	}
	
	/**
	 * Close all file descriptors except:
	 * 
	 *   - 0:   stdin
	 *   - 1:   stdout
	 *   - 2:   stderr
	 *   - 3:   Control socket.
	 *   - dfd: Directory stream (currently opened to /proc/self/fd)
	 */
	
	while((dent = readdir(d))){
		efd = strtol(dent->d_name, &s, 10);
		if(*s != '\0'){/* Rejecting non-integer entries, like . and .. */
			continue;
		}
		if(efd >= 3 && efd != CTRL_SOCK && efd != dfd){
			printf("close(%d)\n", efd);
			close(efd);
		}
	}
	
	/**
	 * Close directory stream. Closes underlying descriptor (dfd) as well.
	 */
	
	closedir(d);
}

/**
 * @brief Double buffer size.
 */

int  doubleBuf(struct iovec* iov){
	iov->iov_len   = !iov->iov_len  ? 1024 : iov->iov_len*2;
	iov->iov_base  = realloc(iov->iov_base,  iov->iov_len);
	return !!iov->iov_base;
}

/**
 * @brief Run Fork Server.
 * 
 * The fork server expects that on FD 3 there will be one end of a UNIX-domain,
 * SOCK_SEQPACKET-type socket.
 * 
 * This function never returns.
 */

BENZINA_PUBLIC void         benzinaForkServerMain       (int argc, char* argv[]){
	struct iovec    iov     = {NULL, 0}, ciov=iov, buf=iov, cbuf=iov;
	struct msghdr   msg     = {NULL,                         /* Optional Address */
	                           sizeof(struct sockaddr_un),   /* Size of Address */
	                           &iov,                         /* Array of scatter/gather buffers */
	                           1,                            /* #     of scatter/gather buffers */
	                           ciov.iov_base,                /* Control message buffer */
	                           ciov.iov_len,                 /* Size of control message buffer */
	                           0};                           /* Flags */
	int             msgSock = -1;
	ssize_t         msgLen  = -1;
	(void)msgSock;
	(void)argc;
	(void)argv;
	
	
	closeallfds();
	
	
	while(1){
		/* 1. Peek-read message. Resize buffers as needed. */
		do{
			iov                = buf;
			ciov               = cbuf;
			msg.msg_name       = NULL;
			msg.msg_namelen    = sizeof(struct sockaddr_un);
			msg.msg_iov        = &iov;
			msg.msg_iovlen     = 1;
			msg.msg_control    = ciov.iov_base;
			msg.msg_controllen = ciov.iov_len;
			msg.msg_flags      = 0;
			msgLen             = recvmsg(CTRL_SOCK, &msg, MSG_PEEK);
			if      (msgLen == 0){
				exit(EXIT_SUCCESS);
			}else if(msgLen  < 0){
				perror("recvmsg");
				exit(EXIT_BADMSG);
			}
			if((msg.msg_flags & MSG_TRUNC ) && !doubleBuf(&buf)){
				exit(EXIT_BADALLOC);
			}
			if((msg.msg_flags & MSG_CTRUNC) && !doubleBuf(&cbuf)){
				exit(EXIT_BADALLOC);
			}
		}while(msg.msg_flags & (MSG_TRUNC|MSG_CTRUNC));
		
		/* 2. True-read message. */
		iov                = buf;
		ciov               = cbuf;
		msg.msg_name       = NULL;
		msg.msg_namelen    = sizeof(struct sockaddr_un);
		msg.msg_iov        = &iov;
		msg.msg_iovlen     = 1;
		msg.msg_control    = ciov.iov_base;
		msg.msg_controllen = ciov.iov_len;
		msg.msg_flags      = 0;
		msgLen             = recvmsg(CTRL_SOCK, &msg, 0);
		if      (msgLen == 0){
			exit(EXIT_SUCCESS);
		}else if(msgLen  < 0){
			exit(EXIT_BADMSG);
		}
		
		/* 3. Parse chdir()/execve() specification from message. */
	}
	
	/* Unreachable */
	exit(EXIT_SUCCESS);
}
BENZINA_PUBLIC int          benzinaForkServerDoLaunch   (const char* forkserverpath){
	(void)forkserverpath;
	return 0;
}
BENZINA_PUBLIC int          benzinaForkServerIsFailed   (void){
	return 0;
}
BENZINA_PUBLIC int          benzinaForkServerSpawn      (const char* filename,
                                                         const char* cwd,
                                                         const char* argv[],
                                                         const char* envp[],
                                                         uint64_t    token){
	(void)filename;
	(void)cwd;
	(void)argv;
	(void)envp;
	(void)token;
	
	int             socks[2];
	pid_t           childPid;
	int             childStatus = 0;
	
	if(socketpair(AF_UNIX, SOCK_SEQPACKET|SOCK_CLOEXEC, 0, socks) < 0){
		return -errno;
	}
	
	childPid = fork();
	if      (childPid  < 0){
		/* We are in the parent; fork() failed. */
		return -errno;
	}else if(childPid == 0){
		/* We are in the child;  fork() succeeded. */
		const char* path  = "./benzinaforkserver";
		char* childArgv[] = {(char*)path, NULL};
		
		close(socks[1]);
		if(socks[0] != CTRL_SOCK){
			dup2(socks[0], CTRL_SOCK);
			close(socks[0]);
		}
		fcntl(CTRL_SOCK, F_SETFD, 0);
		printf("Executing!\n");
		execv(path, childArgv);
		printf("Failed to execv()!\n");
		return -errno;
	}else if(childPid  > 0){
		/* We are in the parent; fork() succeeded. childPid is the PID of the child. */
		close(socks[0]);
		close(socks[1]);
		waitpid(childPid, &childStatus, 0);
		printf("Child exited with status 0x%x!\n", childStatus);
		if      (WIFEXITED  (childStatus)){
			printf("Exited with exit code %d\n", WEXITSTATUS(childStatus));
		}else if(WIFSIGNALED(childStatus)){
			printf("Signalled with signal %d\n", WTERMSIG   (childStatus));
		}else if(WIFSTOPPED (childStatus)){
			printf("Stopped with signal %d\n",   WSTOPSIG   (childStatus));
		}
		return -errno;
	}
	
	return 0;
}


