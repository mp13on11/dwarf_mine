#!/bin/bash

git filter-branch -f --commit-filter '
    GIT_AUTHOR_NAME="mp13on11";
    GIT_AUTHOR_EMAIL="";
    GIT_COMMITTER_NAME="mp13on11";
    GIT_COMMITTER_EMAIL="";
    git commit-tree "$@";
    ' HEAD
