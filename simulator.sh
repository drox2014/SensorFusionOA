rm logs/*
source zamia/path.sh

python sys_monitor.py &
pid=$!
sleep 5

#python main.py &
#
#sleep 15

trap INT "pkill -f sys_monitor.py; exit 1"
#pkill -SIGINT -f main.py
#sleep 5
#pkill -f main.py