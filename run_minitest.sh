echo "Starting..."

MAPS=(Cov_7x7_4Boxes Cov_11x11_9Boxes)
TIME=(250000 300000)

# Special properties
RNN_DIMS=(64 128)

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=false t_max=${TIME[$ind]} rnn_hidden_dim=${RNN_DIMS[$ind]}
    
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=false t_max=${TIME[$ind]} rnn_hidden_dim=${RNN_DIMS[$ind]}

    # echo "Run DCG in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    # python src/main.py -d --config=dcg --env-config=adv_coverage with  \
    #  env_args.map=${MAPS[$ind]} env_args.observe_ids=False obs_agent_id=True obs_last_action=True t_max=${TIME[$ind]} use_cuda=False
done

echo "Run PSeq in Cov_7x7_Safe_Divided for 750000 steps"
python src/main.py -d --config=pseq --env-config=adv_coverage with \
    env_args.map=Cov_7x7_Safe_Divided env_args.observe_ids=True obs_agent_id=False obs_last_action=false t_max=750000 rnn_hidden_dim=64

# RUN OVER AGAIN IN CASE OF EARLY FINISH
for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=false t_max=${TIME[$ind]} rnn_hidden_dim=${RNN_DIMS[$ind]}
    
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=false t_max=${TIME[$ind]} rnn_hidden_dim=${RNN_DIMS[$ind]}
done

for ind in ${!MAPS[*]}
do
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=false t_max=${TIME[$ind]} rnn_hidden_dim=${RNN_DIMS[$ind]}
    
    echo "Run PSeq in ${MAPS[$ind]} for ${TIME[$ind]} steps"
    python src/main.py -d --config=pseq --env-config=adv_coverage with \
     env_args.map=${MAPS[$ind]} env_args.observe_ids=True obs_agent_id=False obs_last_action=false t_max=${TIME[$ind]} rnn_hidden_dim=${RNN_DIMS[$ind]}
done