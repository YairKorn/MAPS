echo "Starting..."

MAPS=(Cov_11x11_9Rooms Cov_11x11_9Rooms_2A)
TIME=(500000 500000)

# Special properties
RNN_DIMS=(128)

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]}
    
    echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=dcg --env-config=adv_coverage with  \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True t_max=${TIME[$ind]} save_model_interval=2000000
done

# Again!
for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]}
    
    echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=dcg --env-config=adv_coverage with  \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True t_max=${TIME[$ind]} save_model_interval=2000000
done