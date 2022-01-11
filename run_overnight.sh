echo "Starting..."

MAPS=(Cov_8x8_Safe_Divided Cov_8x8_Safe_Tunnel Cov_7x7_Safe_Comfort Cov_7x7_Safe_Divided Cov_7x7_Safe_Sparse Cov_7x7_Safe_Tunnel)
TIME=(1250000 1250000 750000 750000 750000 750000)
echo ${MAPS[0]}

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=multi_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=False t_max=${TIME[$ind]}
    
    echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=dcg --env-config=multi_coverage with  \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True obs_last_action=True t_max=${TIME[$ind]}
done

# Again in case it was finished early :)
for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=multi_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=False t_max=${TIME[$ind]}
    
    echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=dcg --env-config=multi_coverage with  \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True obs_last_action=True t_max=${TIME[$ind]}
done

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=multi_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=False t_max=${TIME[$ind]}
    
    echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=dcg --env-config=multi_coverage with  \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True obs_last_action=True t_max=${TIME[$ind]}
done
