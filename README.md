# wordle-ppo
Extends the general concept from [here](https://github.com/andrewkho/wordle-solver/tree/4495ae13ca31ae0f9784b847e34d7ef4117a1819) to solve Wordle with PPO (of course I could've just used an entropy drop heuristic to solve it in like 70 lines but where's the fun in that). Base PPO code from [here](arena3-chapter2-rl.streamlit.app) (the PPO chapter) but moderately modified for Wordle. This was originally developed entirely in a single Jupyter Notebook with no thought for quality, proper documentation, or modularity, so the code will be messy and convuluted until cleaned up once all the additional polishing touches (below) are implemented. 

Major changes from Andrew Ho:
- Uses PPO instead of A2C
- Richer rewards to circumvent the reward sparsity issues of Wordle and improve training
- Learnable word embeddings
- Behavior cloning from entropy policy right off the bat

Changes to make to this implementation:
- Explore tradeoff between less rich state w/ transformer or recurrent block (given that language is sequential). Or, just project down to a lower dimensional state before dot product with decoder/ use a sparse autoencoder
- Cut out as many heuristics as possible, especially in terms of encoding state
- Generally clean up into multiple different files

<figure class="video_container">
  <iframe src="demo.mp4" frameborder="0" allowfullscreen="true"> 
</iframe>
</figure>
