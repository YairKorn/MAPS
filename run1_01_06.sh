echo "Starting..."

# Cov_7x7_Safe_Comfort
python src/main.py -d --config=pseq --env-config=multi_coverage with env_args.map="Cov_7x7_Safe_Comfort" env_args.observe_ids=True obs_agent_id=False obs_last_action=False
python src/main.py -d --config=dcg --env-config=multi_coverage with env_args.map="Cov_7x7_Safe_Comfort" env_args.observe_ids=False obs_agent_id=True obs_last_action=True

# Cov_7x7_Safe_Divided
python src/main.py -d --config=pseq --env-config=multi_coverage with env_args.map="Cov_7x7_Safe_Divided" env_args.observe_ids=True obs_agent_id=False obs_last_action=False
python src/main.py -d --config=dcg --env-config=multi_coverage with env_args.map="Cov_7x7_Safe_Divided" env_args.observe_ids=False obs_agent_id=True obs_last_action=True

# Cov_7x7_Safe_Sparse
python src/main.py -d --config=pseq --env-config=multi_coverage with env_args.map="Cov_7x7_Safe_Sparse" env_args.observe_ids=True obs_agent_id=False obs_last_action=False
python src/main.py -d --config=dcg --env-config=multi_coverage with env_args.map="Cov_7x7_Safe_Sparse" env_args.observe_ids=False obs_agent_id=True obs_last_action=True

# Cov_7x7_Safe_Tunnel
python src/main.py -d --config=pseq --env-config=multi_coverage with env_args.map="Cov_7x7_Safe_Tunnel" env_args.observe_ids=True obs_agent_id=False obs_last_action=False
python src/main.py -d --config=dcg --env-config=multi_coverage with env_args.map="Cov_7x7_Safe_Tunnel" env_args.observe_ids=False obs_agent_id=True obs_last_action=True