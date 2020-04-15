#!/bin/bash

################################
######## LIST SESSIONS #########
################################

SESSIONS=( $(
    tmux list-sessions |
    while read -r line
    do 
        echo "$line" | sed 's/:.*//'
    done
    ))

#echo "${SESSIONS[@]}"

length=${#SESSIONS[@]}
echo "#running sessions: $length"

for ((i = 0 ; i < $length ; i++)); do
  echo "[$i] ${SESSIONS[$i]}"
done

################################
######## SELECT SESSION ########
################################

echo "Select session to load: "
read session_id
echo "Selected session: ${SESSIONS[$session_id]}"
tmux attach-session -t ${SESSIONS[$session_id]}


################################
######### NEEW SESSION #########
################################
#tmux new -s my_session





################################
############ TESTING ###########
################################
# declare -a NAME # explicitly declare array
# counter=0

# SESSIONS=( $(
# tmux list-sessions |
# while read -r line
# do
#     echo "$line"
#     #echo "["$counter"] $line"
#     #NAME[$counter]="$(echo "$line" | sed 's/:.*//')"
#     #NAME+=$(echo "$line" | sed 's/:.*//')
#     #my_array+=("$line")
#     #my_array[$counter]="value2"
#     #counter=$((counter+1))
# done
# ))

# echo "Select session to load: "
# #read session_id
# #echo "Selected session ${NAME[$session_id]}"

# length=${#NAME[@]}
# echo "Running Sessions: $length"

# my_array+=(baz)
# echo "${my_array[@]}"
# echo "${SESSIONS[@]}"