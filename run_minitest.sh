echo "Starting..."

MAPS=(Threat_6x6_Random Threat_7x7_Hard Threat_8x8_Hard) # Threat_5x5_2Path Threat_5x5_NoSol Threat_5x5_Sparse Threat_6x6_Medium Threat_6x6_Random)
TIME=(500000 750000 1000000) # 300000 300000 300000 600000 600000)

for ind in ${!MAPS[*]}
do
    echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=dcg --env-config=adv_coverage with  \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True t_max=${TIME[$ind]}

    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]} save_model=True
    
done

or ind in ${!MAPS[*]}
do
    echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=dcg --env-config=adv_coverage with  \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True t_max=${TIME[$ind]}
     
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]} save_model=True
    
done