# MAPS=(Threat_5x5_Hard Threat_6x6_Medium Threat_6x6_Hard Threat_7x7_Hard)
# TIME=(300000 500000 600000 800000)

# for i in {1..1}; do
#     echo "Iteration number $i:"

#     for ind in ${!MAPS[*]}; do
#         echo "Run MAPS in ${MAPS[$ind]} for ${TIME[$ind]} steps"
#         python src/main.py --config=maps --env-config=adv_coverage with \
#         env_args.map=${MAPS[$ind]} obs_agent_id=False t_max=${TIME[$ind]} save_model=True #save_model_interva=5000

        # echo "Run tabular DQL in ${MAPS[$ind]} for ${TIME[$ind]} steps"
        # python src/main.py --config=iql_tabular --env-config=hunt_trip with \
        # obs_agent_id=False t_max=${TIME[$ind]} env_args.map=${MAPS[$ind]} epsilon_anneal_time=300000 #env_args.catch_validity=True

#     done
# done

# MAPS=(Threat_5x5_SingleH Threat_6x6_SingleH) #Hunt_7x7_Multi
# TIME=(600000 900000) #100000

# for ind in ${!MAPS[*]}; do
#     SIM_DECAY=(0.0) # 1e-8 0.2 0.5 0.8 

#     for u in ${!SIM_DECAY[*]}; do
#         echo "Run tabular MAPS in ${MAPS[$ind]} for ${TIME[$ind]} steps with and DECAY=${SIM_DECAY[$u]}"
#         python src/main.py --config=maps_tabular --env-config=adv_coverage with obs_agent_id=False \
#         t_max=${TIME[$ind]} env_args.map=${MAPS[$ind]} epsilon_anneal_time=400000 env_args.simulation_decay=${SIM_DECAY[$u]}

#     done
# done


MAPS=(S5x5 T5x5_SingleH S7x7_4Boxes S11x11_9Boxes)
TIME=(400000 800000 1200000 2000000)

for i in {1..1}; do
    echo "Iteration number $i:"

    for ind in ${!MAPS[*]}; do
        echo "Run MAPS in ${MAPS[$ind]} for ${TIME[$ind]} steps"
        python src/main.py --config=maps_tabular --env-config=adv_coverage with \
        env_args.map=${MAPS[$ind]} obs_agent_id=False t_max=${TIME[$ind]} epsilon_anneal_time=400000 save_model=True #save_model_interva=5000

    done
done