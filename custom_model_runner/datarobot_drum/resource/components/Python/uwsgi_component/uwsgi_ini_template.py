WSGI_INI_CONTENT = """
[uwsgi]

# If VIRTAL_ENV is set then use its value to specify the virtualenv directory
if-env = VIRTUAL_ENV
print = Your virtualenv is %(_)
virtualenv = %(_)
endif =

# placeholders that you have to change
restful_app_folder = {restful_app_folder}
# logs_folder = %(restful_app_folder)/logs

my_user = root

# exec-asap = mkdir -p %(logs_folder)
# daemonize uWSGI after app loading. Note that this option is mandatory
# and should not be changed! See comment in 'uwsgi_broker.py'
# daemonize2 = %(logs_folder)/uwsgi-@(exec://date +%%Y-%%m-%%d).log
# logto = %(logs_folder)/uwsgi-@(exec://date +%%Y-%%m-%%d).log

{comment_out_master}logger = {log_socket}
{comment_out_non_master}logto = {logto}

# exit if no app can be loaded
need-app = true

disable-logging = {disable_logging}

# log max size is 1MB.
# log-maxsize = 1048576
# log-reopen = true

#Cron:
# - check if logs size bigger then 10 MB:
#       du -sb /tmp/restful_comp_TbVOX1/logs | [[ $(awk '{{print $1}}') > 70000 ]] && echo bigger
# - delete 10 oldest files:
#       find /tmp/restful_comp_TbVOX1/logs  -type f -printf '%Ts\\t%p\\n' | \
#       sort -nr | cut -f2 | tail -n 10 | xargs -n 1 -I '_' rm _

pidfile = %(restful_app_folder)/{pid_filename}
socket = %(restful_app_folder)/{sock_filename}
chdir = %(restful_app_folder)
file = {restful_app_file}
callable = {callable_app}

# environment variables
env = CUDA_VISIBLE_DEVICES=-1
env = PYTHONPATH=%(restful_app_folder):{python_paths}

master = {master}
log-master = {master}
processes = {workers}

# allows nginx (and all users) to read and write on this socket
chmod-socket = 666

# remove the socket when the process stops
vacuum = true

# loads your application one time per worker
# will very probably consume more memory,
# but will run in a more consistent and clean environment.
lazy-apps = false

uid = %U
gid = %G

# uWSGI will kill the process instead of reloading it
die-on-term = true

# socket file for getting stats about the workers
stats = %(restful_app_folder)/{stats_sock_filename}

# Cheaper memory limits
# cheaper-rss-limit-soft - soft limit will prevent cheaper from spawning new workers
# if workers total rss memory is equal or higher
#
# cheaper-rss-limit-hard limit will force cheaper to cheap single worker
# if workers total rss memory is equal or higher
{cheaper_memory_limits}

# Scaling the server with the Cheaper subsystem

# set cheaper algorithm to use, if not set default will be used
cheaper-algo = spare

# minimum number of workers to keep at all times
{comment_out_cheaper}cheaper = {cheaper}

# number of workers to spawn at startup
{comment_out_cheaper}cheaper-initial = {cheaper_initial}

# maximum number of workers that can be spawned
{comment_out_cheaper}workers = {workers}

# how many workers should be spawned at a time
{comment_out_cheaper}cheaper-step = {cheaper_step}

# skip atexit hooks (ignored by the master)
skip-atexit = false

# skip atexit teardown (ignored by the master)
skip-atexit-teardown = true

# enable metrics subsystem
enable-metrics = {enable_metrics}

# oid prefix: 1.3.6.1.4.1.35156.17.3
# metric = name=pm-counter1,type=counter,initial_value=0,oid=100.1
{metrics}
"""
