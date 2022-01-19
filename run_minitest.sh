echo "Starting..."

MAPS=(Cov_4x4_Safe Cov_5x5_Safe_Single Cov_5x5_Safe Cov_7x7_4Boxes Cov_11x11_9Boxes)
TIME=(100000 100000 250000 250000 400000)

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]} save_model=True
    
    echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=dcg --env-config=adv_coverage with  \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True t_max=${TIME[$ind]}
done

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]} save_model=True

done

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]} save_model=True

done

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False t_max=${TIME[$ind]} save_model=True

done