# wordle-ppo
Extends the general concept from [here](https://github.com/andrewkho/wordle-solver/tree/4495ae13ca31ae0f9784b847e34d7ef4117a1819) to solve Wordle with PPO. Base PPO code from [here](arena3-chapter2-rl.streamlit.app) (the PPO chapter) but moderately modified for Wordle. This was developed entirely in a single Jupyter Notebook with no thought for quality, proper documentation, or modularity, so the code will be messy and convuluted until cleaned up. 

Major changes from Andrew Ho:
- Uses PPO instead of A2C, obviously
- Richer rewards to circumvent the reward sparsity issues of Wordle and improve training
- Learnable word embeddings
- Behavior cloning from "expert" (entropy drop) policy right off the bat
- Curriculum training all the way up

Changes to make to this implementation:
- Explore less rich state w/ transformer or recurrent block (or just project down to a lower dimensional state before dot product with decoder or just use a sparse autoencoder) 
- Explore playing around with recent fails queue -> didn't help me but it apparently did help Andrew Ho
- Explore whether or not to connect actor and critic instead of keeping them separate -> separate had empirically better results but that may be downstream of other changes because intuitively it feels like a combined network makes more sense here
- Any other ways to cut out some of the heuristics I used based on the game itself while having similar training time -> especially in terms of encoding state 
