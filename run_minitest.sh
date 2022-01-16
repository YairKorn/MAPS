echo "Starting..."

MAPS=(Cov_5x5_Safe_Single Cov_11x11_9Boxes Cov_7x7_Safe_Tunnel Cov_11x11_9Rooms)
TIME=(100000 300000 750000 1250000)

# Special properties
RNN_DIMS=(128)

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]} save_model=True
    
    # echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    # python src/main.py -d --config=dcg --env-config=adv_coverage with  \
    #  env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True t_max=${TIME[$ind]}
done

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]} save_model=True

done