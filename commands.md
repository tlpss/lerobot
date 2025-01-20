# dataset generation
- copy gello pickles to `data/raw`
- `python lerobot/scripts/push_dataset_to_hub.py --raw-dir data/raw/ursquee1 --raw-format ursquee_pkl --repo-id remko/ursquee1 --push-to-hub 0 --video 1 --local-dir data/remko/ursquee1 --force-override 1 --episodes 1`

# visualize dataset
- `python lerobot/scripts/visualize_dataset.py --repo-id remko/ursquee1 --root data/ --episode-index 0`

# train 
- adapt config in `lerobot/config/experiment` 
- `wandb login`
- `export DATA_DIR=/home/tlips/Code/lerobot/data/`
- `python lerobot/scripts/train.py +experiment=ursquee-act`

## usage of 'screen'
Training should not be run from a regular terminal, since the process will stop if the UI is stopped (e.g. upon account lock).
-`screen -ls` to print running processes 
-`screen -S <desired-name-of-process>` to start a process
-Ctrl+A Ctrl+D to detach process
-`screen -R -D <name-of-process-to-resume>` resumes process and detaches others from it (if -D is forgotten, a new process will be created)