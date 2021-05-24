#!/bin/bash
for server in $(cat itesm-2020-students_openstack | cut -d'-' -f1-3);
do for log_file in $(ls | grep $server); do sed -i -e "s|\<itesm-2020-[^ ]*\>|${server} |g" $log_file; done; done
for server in $(ls | grep syslog_log); do cat $server | sed -e 's/^"//' -e 's/"$//' > $server; done
for server in $(ls | grep syslog_log); do mv $server $(echo "$server" | awk -F- '{print $1 FS $2 FS $3 FS "syslog_log"}'); done
