import checkers_rl2.checkers2 as checkers2
import matplotlib.pyplot as plt
from keras import Sequential, regularizers
from keras.layers import Dense
import tensorflow as tfw
import numpy as np
import random
from tqdm import tqdm
import csv

def concatenate(array1, array2):
    for i in range(len(array2)):
        array1.append(array2[i])
    return array1  

def GetModel(Oppenent):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=5)) 
    model.add(Dense(16, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))

    model.add(Dense(1, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["acc"])

    data = [] 
    labels = np.zeros(1)
    winrates = []
    learning_rate = 0.5
    discount_factor = 0.95
    exploration = 0.95
    win = 0
    lose = 0
    draw = 0
    AGENT_DEPTH = 1          # Agent uses depth-1 (DQN with leafs)
    OPPONENT_DEPTH = 1  
    frozen_model = None

    for generations in tqdm(range(2000)):
        if generations % 100 == 0:
            frozen_model = tfw.keras.models.clone_model(model)
            frozen_model.set_weights(model.get_weights())
        data = []
        for g in range(10):
            temp_data = []
            game = checkers2.Checkers()
            player = 1 if g % 2 == 0 else -1  # Alternate: even games agent goes first, odd games opponent goes first
            count = 0
            while True:
                count += 1
                end2 = 0
                if count > 1000 :
                    draw += 1
                    break

                else :
                    if (player == 1) : 
                        leafs = game.minmax(player, depth=AGENT_DEPTH, RL=True)
                        Leaf = tfw.zeros((len(leafs), 5))
                        for l in range(len(leafs)) :
                            tensor = leafs[l][2]
                            Leaf = tfw.tensor_scatter_nd_update(Leaf, [[l]], [tensor[:5]])
                        scores = frozen_model.predict_on_batch(Leaf)
                        if (len(scores) == 0):
                            end2 = -player
                            continue
                        i = np.argmax(scores)
                        game.PushMove(leafs[i][0])
                        tab = leafs[i][2][:5]
                        temp_data.append(tab)
                    elif (player == -1):
                        if Oppenent == "random":
                            leafs = game.GetValidMoves(player) 
                            if (len(leafs) == 0):
                                end2 = -player
                                continue
                            move = random.choice(leafs)
                            game.PushMove(move)

                        elif Oppenent == "minmax":
                            moves = game.minmax(player, depth=OPPONENT_DEPTH)
                            if len(moves) == 0 : 
                                end2 = -player
                                continue
                            if random.random() >= exploration:
                                Moves = game.GetValidMoves(player) 
                                move = random.choice(Moves)
                                game.PushMove(move)
                            else :
                                move = random.choice(moves)
                                game.PushMove(move)
                        
                        elif Oppenent == "itself":
                            leafs = game.minmax(player, depth=OPPONENT_DEPTH, RL=True)
                            Leaf = tfw.zeros((len(leafs), 5))
                            for l in range(len(leafs)) :
                                tensor = leafs[l][2]
                                Leaf = tfw.tensor_scatter_nd_update(Leaf, [[l]], [tensor[:5]])
                            scores = model.predict_on_batch(Leaf)
                            if (len(scores) == 0):
                                end2 = -player
                                continue
                            if (random.random() >= exploration):
                                move = random.choice(leafs)
                                move = move[0]
                                game.PushMove(move)
                            else:
                                i = np.argmax(scores)
                                game.PushMove(leafs[i][0])
                        elif Oppenent == "trained":
                            continue
                        elif Oppenent == "curriculum":   # Let the curriculupponent play with dynamic randomness
                            if not hasattr(game, "curriculum_randomness"):
                                game.curriculum_randomness = 0.8

                            if hasattr(game, "last_result"):
                                if game.last_result == -1:  # agent won last time
                                    game.curriculum_randomness = max(0.1, game.curriculum_randomness - 0.05)
                                elif game.last_result == 1:  # agent lost
                                    game.curriculum_randomness = min(0.9, game.curriculum_randomness + 0.05)

                            leafs = game.GetValidMoves(player)
                            if len(leafs) == 0:
                                end2 = -player
                                continue
                            if random.random() < game.curriculum_randomness:
                                move = random.choice(leafs)
                            else:
                                move = random.choice(game.minmax(player, depth=1))
                            game.PushMove(move)
                        else :
                            raise ValueError(f"Unknown opponent type: {Oppenent}")

                end = game.EndGame()

                if end == 1 or end2 == 1:
                    win += 1
                    reward = 10
                    temp_tensor = tfw.constant(temp_data[1:])
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_futur_value = np.ones(old_prediction.shape)
                    temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_futur_value - old_prediction )
                    data = concatenate(data, temp_data[1:])
                    labels = np.vstack((labels, temp_labels))
                    break
			

                
                elif end == -1 or end2 == -1: 
                    lose = lose + 1
                    reward = -10
                    temp_tensor = tfw.constant(temp_data[1:])
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_futur_value = -1*np.ones(old_prediction.shape)
                    temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_futur_value - old_prediction )
                    data = concatenate(data, temp_data[1:])
                    labels = np.vstack((labels, temp_labels))
                    break

                player = -player 
        data = tfw.constant(data)
        model.fit(data[1:], labels[2:], epochs=16, batch_size=256, verbose=0)
        labels = np.zeros(1)
        generation_stats = {
            "generation": generations,
            "win": win,
            "loss": lose,
            "draw": draw,
            "winrate": int((win) / (win + draw + lose) * 100),
            "opponent_winrate": int((lose) / (win + draw + lose) * 100),
            "drawrate": int((draw) / (win + draw + lose) * 100)
        }
        winrates.append(generation_stats)

        # Reset for next generation
        win = 0
        lose = 0
        draw = 0

        model.save("models/"+Oppenent+"aa.keras")

    
    with open(f"winrates_{Oppenent}.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "win", "loss", "draw", "winrate", "opponent_winrate", "drawrate"])
        writer.writeheader()
        writer.writerows(winrates)


    indices = list(range(len(winrates)))
    agent_winrate = [x["winrate"] for x in winrates]
    opponent_winrate = [x["opponent_winrate"] for x in winrates]

    def average_every_n(lst, n):
        return [sum(lst[i:i+n]) / len(lst[i:i+n]) for i in range(0, len(lst), n)]

# Smooth over every 50 generations
    n = 50
    agent_winrate = [x["winrate"] for x in winrates]
    opponent_winrate = [x["opponent_winrate"] for x in winrates]
    avg_agent_winrate = average_every_n(agent_winrate, n)
    avg_opponent_winrate = average_every_n(opponent_winrate, n)
    avg_indices = list(range(0, len(agent_winrate), n))

    # Plot smoothed winrates
    plt.figure(figsize=(10, 6))
    plt.plot(avg_indices, avg_agent_winrate, label="Trained Agent Win Rate (avg/50)", linestyle='-')
    plt.plot(avg_indices, avg_opponent_winrate, label=f"{Oppenent.capitalize()} Opponent Win Rate (avg/50)", linestyle='--')

    plt.title("Smoothed Win Rate Progress Over Generations", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if "__main__" == __name__ :
    Oppenent = "curriculum"
    print(Oppenent)
    GetModel(Oppenent=Oppenent)