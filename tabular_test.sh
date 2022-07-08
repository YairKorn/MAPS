# MAPS=(Threat_5x5_Hard Threat_6x6_Medium Threat_6x6_Hard Threat_7x7_Hard)
# TIME=(300000 500000 600000 800000)
MAPS=(Threat_8x8_Hard)
TIME=(1250000)

# for i in {1..1}; do
#     echo "Iteration number $i:"

#     for ind in ${!MAPS[*]}; do
#         echo "Run MAPS in ${MAPS[$ind]} for ${TIME[$ind]} steps"
#         python src/main.py --config=maps --env-config=adv_coverage with \
#         env_args.map=${MAPS[$ind]} obs_agent_id=False t_max=${TIME[$ind]} save_model=True #save_model_interva=5000

#         echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
#         python src/main.py --config=dcg --env-config=adv_coverage with  \
#          env_args.map=${MAPS[$ind]} obs_agent_id=True t_max=${TIME[$ind]}
#     done
# done

MAPS=(Hunt_5x5_Simple Hunt_7x7_Multi) #
TIME=(400000 700000) #100000

for i in {1..1}; do
    echo "Iteration number $i:"

    for ind in ${!MAPS[*]}; do
        echo "Run tabular MAPS in ${MAPS[$ind]} for ${TIME[$ind]} steps"
        python src/main.py --config=maps_tabular --env-config=hunt_trip with \
        obs_agent_id=False t_max=${TIME[$ind]} env_args.map=${MAPS[$ind]} epsilon_anneal_time=200000 #env_args.catch_validity=True

        # echo "Run tabular DQL in ${MAPS[$ind]} for ${TIME[$ind]} steps"
        # python src/main.py --config=iql_tabular --env-config=hunt_trip with \
        # obs_agent_id=False t_max=${TIME[$ind]} env_args.map=${MAPS[$ind]} epsilon_anneal_time=300000 #env_args.catch_validity=True
    done
done