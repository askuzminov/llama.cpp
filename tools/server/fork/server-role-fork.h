#pragma once

// fork: the router child side of the fork - stdin reader, stop timeout, exit and memory wait. Used by
// server_role (server-common.cpp), so it lives in the server-context library with it

#include <string>

// not std::cin: a thread blocked in it holds the stdin FILE lock, and the CRT takes that lock at exit
bool server_child_read_stdin(std::string & buf);

// the router force-kills a child that is still alive stop_timeout seconds after the exit command.
// If the router dies before that (Windows kills it 5 s after its console window is closed), the
// child applies the same limit itself when its stdin closes
void server_child_watch_router_after_exit(int stop_timeout);

// LLAMA_SERVER_STOP_TIMEOUT, set by the router for its children (DEFAULT_STOP_TIMEOUT of server-models.cpp when unset)
int server_child_stop_timeout();
