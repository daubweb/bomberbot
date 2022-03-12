# Ideen

## General
* Only a window around the bot as features
* Getting Reward via Events
* Decreasing epsilon (probability of a random action)
* Place Bombs only if you can escape its blastrange
## Algorithms
* Monte Carlo


## Training Optimisation
* Run some games and record game state, rewards and actions to create some baseline-data
* Positions of walls are always known, so you can be sure to always select a valid action
* Train against yourself
* 

## Features
* Game State:
  * Positions of Bombs
    * Is a bomb next to me?
  * Positions of Coins
  * Positions of Opponents

* Only n fields next to agent
* Score
* Score and Position of Opponents --> Could lead to more or less aggressive behaviour of agent
* Is a bomb nearby? --> Escape the bombs! 


## Game State
| Feature | Interesting |Note|
|---------|-------------|---|
|Positions of enemies | YES | Aggressiveness?
|Score of enemies | NO | Not that interesting
| Board-State | ONLY_IN_AREA | What are the fields near me?
|Positions of bombs | ONLY_IN_BLAST_RANGE | Crucial to survival
|Positions of coins | ONLY_IN_AREA | Only the
| My Score | NO | Not that interesting, as a good reward leads to a good score, maybe you could use the mean score to determine a strategy