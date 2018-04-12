#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
. activate znymky
echo "$USER running on $HOSTNAME: $@"
exec $@ 2>/dev/null
