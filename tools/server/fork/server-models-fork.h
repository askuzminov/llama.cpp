#pragma once

// fork: router child helpers used by server_child::setup. The memory wait of a child spawned
// alone and the rest of the fork's server_models / server_child members are in server-models-fork.cpp

#include <string>

// not std::cin: a thread blocked in it holds the stdin FILE lock, and the CRT takes that lock at exit
bool server_child_read_stdin(std::string & buf);

// the router force-kills a child that is still alive stop_timeout seconds after the exit command.
// If the router dies before that (Windows kills it 5 s after its console window is closed), the
// child applies the same limit itself when its stdin closes
void server_child_watch_router_after_exit(int stop_timeout);
