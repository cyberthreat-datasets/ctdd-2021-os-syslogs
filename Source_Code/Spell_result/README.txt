The following directories contain the original log files containing normal and abnormal logs.
They also contain the generated templates and the structured log file containing teh substituted template.
The last number in each file indicate the tau value used in spell for that particular parse operation.


# To get how many logs or templates we have ina  scpecific file use:
wc -l <file_name>

# To view templates in template files sorted by length use the following:
awk -F',' '{printf "%d %s\n", length($2), $2}'  linux.log_templates.csv | sort -n -k1 -k2
