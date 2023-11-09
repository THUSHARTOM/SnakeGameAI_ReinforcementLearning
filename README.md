
# Reinforcement Learning        
- Reference - Python Engineer

 >*Reinforcement  learning is teaching a software to behave in an environment based on how good it is doing.*

Here we use Deep Q Learning which is an appraoch that extends reinforcement learning by using a deep neural network to predict the actions.

## Theory

Our project consists of 3 main parts
- **Game (using Pygame)**
    - play_step(action)
        - reward, game_over, score
- **A Model** (using PyTorch)
    - Linear_QNet (DQN) [Feed forward neural network]
        - model.predict(state) -> action
- **An Agent** (using both the above)
    - Game
    - model <br>
    *Training*
    - state = get_state(game)
    - action = get_move(state):
        - model.predict()
    - reward, game_over. score = game.play_step(action)
    - new_state = get_state(game)
    - remember
    - model.train()

    ### Rewards
    - eat food = +10
    - game over = -10
    - else = 0
    ### Actions
    - [1,0,0] -> Straight
    - [0,1,0] -> right turn
    - [0,0,1] -> left turn 
    <br>
    _Cannot go 180` turn this way_

    ### state(11 values)
    [danger straight, danger right, danger left,
    
    direction left, direction right,
    direction up, direction down,
    
    food left, food right,
    food up, food down
    ]

    ### Model
    ![](./Assets/model.png "model")

    #### Deep Q learning
    Q value = Quality of action

    0. Init Q value(=init model)
    1. choose action (model.predict(state)) <-> or a random move
    2. Perform action
    3. Measure reward
    4. Update Q value (+train model)

    > 4->1->4->1... Iterative learning step

    ### Bellman Equation
    ![](./Assets/Bellman.png "Bellman equation")

    **Loss function** - Mean squared error

    loss = (Q<sub>new</sub> - Q)<sup>2</sup>

## Implementat the Game

**Requirements**
- Pytorch (No cuda)
```
pip3 install torch torchvision
```
- pygame
```
pip install pygame
```
- Snake game GitHub repo (by Python Engineer)
```
https://github.com/patrickloeber/python-fun/tree/master/snake-pygame
```

**Approach**
1. Create a reset function to reset all the parameters once the game is lost
2. Create a variable frame_iteration to keep track of the frame iteration. 
3. Change move()
    - replace the human input *direction* arguement by *action* agent input
    - To move the snake we design an algorithm to move straight, turn right and left
4. Change play_step() by 
    - adding a reward parameter [0,+10,-10]
    - To make sure the snake dont keep increasing in size indefinitly we cap it at 100*len(snake) using frame_iteration. Add +1 to the frame_iteration every time the function gets executed
    - Also return reward along with game_over state and score
5. Change _is_collision()
    - To give the AI some context about the environment, we create a point pt which keeps track of hitting the boudaries or on itself

## Implement the agent
## Implement the model
 
