/* Includes */
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "linenoise.h"
#include "encodings/utf8.h"
#include "benzina/benzina.h"
#include "internal.h"



/* Defines */
#define WHITESPACECHARS   " \f\n\r\t\v"
#define BENZ_SHELL_PROMPT "B$ "
#define BENZ_HISTORY_FILE ".benz_history"




/* Static Functions */
BENZINA_STATIC int match_cmd(const char* buf, const char* cmd){
    while(*cmd){
        switch(*cmd){
            case '*': return 1;
            case '$': return !buf[strspn(buf, WHITESPACECHARS)];
            case '^': while(isspace(*buf)){buf++;} cmd++; break;
            case ' ':
                if(!isspace(*buf))
                    return 0;
                while(isspace(*++buf));
                while(*++cmd == ' ');
            break;
            default:
                if(*buf++ != *cmd++)
                    return 0;
        }
    }
    return 1;
}

/**
 * @brief Set up command-line history tracking in $HOME/.benz_history
 * @param [out] histpath  A pointer to the path of the history file being opened,
 *                        $HOME/.benz_history.
 */

BENZINA_STATIC void setup_history(char** histpath){
    size_t histlen;
    char* HOME = getenv("HOME");
    if(!HOME || !*HOME)
        return;
    
    histlen = snprintf(*histpath, 0, "%s/%s", HOME, BENZ_HISTORY_FILE)+1;
    *histpath = malloc(histlen);
    if(!*histpath){
        fprintf(stderr, "Out of memory!\n");
        exit(1);
    }
    snprintf(*histpath, histlen, "%s/%s", HOME, BENZ_HISTORY_FILE);
    linenoiseHistorySetMaxLen(1000);
    linenoiseHistoryLoad(*histpath);
}

/**
 * @brief Tear down command-line history tracking.
 * @param [out]  histpath  A pointer to the path of the history file being closed,
 *                         $HOME/.benz_history.
 */

BENZINA_STATIC void teardown_history(char** histpath){
    linenoiseHistorySave(*histpath);
    free(*histpath);
}

/**
 * @brief Propose completions for a user's currently-typed command on TAB.
 * @param [in]     buf  The current line of user input.
 * @param [in,out] lc   The vector of line completions.
 */

BENZINA_STATIC void completer(const char* buf, linenoiseCompletions* lc){
    if      (match_cmd(buf, "^bcachefs show*")){
        linenoiseAddCompletion(lc, "bcachefs show super ");
    }else if(match_cmd(buf, "^bcachefs*")){
        linenoiseAddCompletion(lc, "bcachefs show ");
    }else if(match_cmd(buf, "^b*")){
        linenoiseAddCompletion(lc, "bcachefs ");
    }else if(match_cmd(buf, "^c*")){
        linenoiseAddCompletion(lc, "cd ");
    }else if(match_cmd(buf, "^e*")){
        linenoiseAddCompletion(lc, "exit ");
    }else if(match_cmd(buf, "^h*")){
        linenoiseAddCompletion(lc, "hexdump ");
    }else if(match_cmd(buf, "^l*")){
        linenoiseAddCompletion(lc, "ls ");
    }else if(match_cmd(buf, "^o*")){
        linenoiseAddCompletion(lc, "open ");
    }else if(match_cmd(buf, "^$")){
        linenoiseAddCompletion(lc, "bcachefs ");
        linenoiseAddCompletion(lc, "cd ");
        linenoiseAddCompletion(lc, "exit ");
        linenoiseAddCompletion(lc, "hexdump ");
        linenoiseAddCompletion(lc, "ls ");
        linenoiseAddCompletion(lc, "open ");
    }
}

/**
 * @brief Propose hints for a user's currently-typed command.
 * @param [in]  buf    The current line of user input.
 * @param [out] color  The color of the hint (31-37 corresponding to ANSI colors).
 *                     | red | green | yellow | blue | magenta | cyan | white |
 *                     | 31  |  32   |   33   |  34  |    35   |  36  |   37  |
 * @param [out] bold   Whether to make the hint bold (!0) or not (0).
 * @return The hint-string.
 */

BENZINA_STATIC char* hinter(const char* buf, int* color, int* bold){
    if     (match_cmd(buf, "^exit$"))
        return *color=34, *bold=1, " [exitcode?]";
    else if(match_cmd(buf, "^open$"))
        return *color=34, *bold=1, " <disk image>";
    else if(match_cmd(buf, "^ls$"))
        return *color=34, *bold=0, " [directory]";
    else if(match_cmd(buf, "^cd$"))
        return *color=34, *bold=0, " <directory>";
    else if(match_cmd(buf, "^hexdump$"))
        return *color=34, *bold=0, " <from> <to>";
    else if(match_cmd(buf, "^bcachefs$"))
        return *color=34, *bold=0, " show";
    else if(match_cmd(buf, "^bcachefs show$"))
        return *color=34, *bold=0, " super";
    
    return NULL;
}

/**
 * @brief Evaluate one REPL command line.
 * 
 * May alter the input line of text in any way desired.
 * 
 * @param [in,out] buf          The full line of user input.
 * @param [out]    exitcode     The desired exit code, if exiting.
 * @return Whether the application should exit (!0) or not (0).
 */

BENZINA_STATIC int   eval_line(char* buf, int* exitcode){
    char* linez = buf+strspn(buf, WHITESPACECHARS);
    if(!*linez)
        return 0;
    
    if      (match_cmd(buf, "^exit$")){
        *exitcode = EXIT_SUCCESS;
        return 1;
    }else if(match_cmd(buf, "^exit *")){
        sscanf(buf, " exit %d", exitcode);
        return 1;
    }
    
    return 0;
}



/**
 * @brief BcacheFS Shell
 * 
 * Implements a small, shell-like environment for exploration of BcacheFS
 * disk images.
 * 
 * @param [in] argc  The classic main() argument count  mandated by Standard C.
 * @param [in] argv  The classic main() argument vector mandated by Standard C.
 * @return Exit code.
 */

BENZINA_STATIC int main_bcachefs_shell(int argc, char* argv[]){
    (void)argc;
    (void)argv;
    char*     line;
    char*     histpath=NULL;
    int       exiting, exitcode=EXIT_SUCCESS;
    const int interactive=isatty(STDIN_FILENO);
    
    
    linenoiseSetMultiLine(1);
    linenoiseSetEncodingFunctions(linenoiseUtf8PrevCharLen,
                                  linenoiseUtf8NextCharLen,
                                  linenoiseUtf8ReadCode);
    if(interactive){
        setup_history(&histpath);
        linenoiseSetCompletionCallback(completer);
        linenoiseSetHintsCallback(hinter);
        linenoiseSetFreeHintsCallback(NULL);
    }
    
    for(exiting=0; !exiting && (line=linenoise(BENZ_SHELL_PROMPT)); linenoiseFree(line)){
        if(interactive && strspn(line, WHITESPACECHARS)[line])
            linenoiseHistoryAdd(line);
        
        exiting = eval_line(line, &exitcode);
    }
    
    if(interactive)
        teardown_history(&histpath);
    
    return exitcode;
}

BENZINA_TOOL_REGISTER("shell", main_bcachefs_shell)
