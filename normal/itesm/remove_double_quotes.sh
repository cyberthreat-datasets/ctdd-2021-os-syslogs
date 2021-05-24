#!/bin/bash
for server in $(ls | grep syslog_log); do cat $server | sed -e 's/^"//' -e 's/"$//' > $server-temp; done
for server in $(ls | grep syslog_log); do mv $server $(echo "$server" | awk -F- '{print $1 FS $2 FS $3 FS "syslog_log"}'); done
