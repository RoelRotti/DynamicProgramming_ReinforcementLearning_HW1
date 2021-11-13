import math
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
plt.style.use("seaborn")

def homework_one(question):

    f = [500, 300, 200]
    C = 100
    T = 600

    C = C + 1 # so rows/columns with zeros accessable
    T = T + 2

    def lambda_t(t, i):
        mu = [0.001, 0.015, 0.05]
        vu = [0.01, 0.005, 0.0025]
        return mu[i] * np.exp(vu[i]*t)

    def value_function(action, x, t):
        # Person: 500 (0), 300 (1), 200 (2)
        if action == 500:
            willing_to_buy = lambda_t(t, 0)
        elif action == 300:
            willing_to_buy = lambda_t(t, 0) + lambda_t(t, 1) 
        elif action == 200:
            willing_to_buy = lambda_t(t, 0) + lambda_t(t, 1) + lambda_t(t, 2)

        value = willing_to_buy * (action + V[x-1, t+1]) + (1 - willing_to_buy) * V[x, t+1]
        return (value, willing_to_buy)

    # define matrix value function
    V = np.zeros((C, T))
    # define matrix optimal policy
    optimal_policy = np.zeros((C, T))

    # Loop backwards through time
    for t in range(T-2, 0, -1):
        # loop over states
        for x in range(1,C):

            # for each action possible
            value_per_action = []
            for action in f:
                value_per_action.append(value_function(action, x, t)[0])
            max_value = max(value_per_action)
            V[x, t] = max_value
            best_option = f[value_per_action.index(max(value_per_action))]
            optimal_policy[x,t] = best_option

    ## A: Determine the total expected revenue and the optimal policy by implementing dynamic programming yourself in python

    print("A: Total Expected revenue =", V[100, 1])

    ## B: Makes a plot of the policy (with time and capacity on the axes)
    if question == "B":
        # 3. Plot the heatmap
        plt.axes().set_aspect('equal')
        plt.imshow(optimal_policy, alpha=0.8, cmap='YlOrBr_r')
        
        plt.title( "HeatMap of the Optimal Policy" )
        plt.colorbar()
        
        plt.legend()
        plt.show()


    ## C: Simulate the demand over time, and for this realization determine which tickets are sold, what 
    # the remaining capacity is and what the prices are at each moment
    if question == "C":
        # Imagine you start at V[100, 1]
        # Loop through time
        capacity = [] # minus one since earlier 1 is added to allow for better matrices
        capacity.append(C-1)
        total_price = [0] # To let the index start at 1
        for t in range(1, T-2):
            if capacity[-1] == 0:
                break
            price = optimal_policy[capacity[-1], t]
            prob_selling = value_function(price, capacity[-1], t )[1]
            if np.random.random() <= prob_selling: # Check if number between [0,1) is smaller than probability
                capacity.append(capacity[-1] - 1)
                total_price.append(price)
            else:
                capacity.append(capacity[-1])
                total_price.append(0)


        cumulative_price = np.cumsum(total_price)

        print(len(total_price))
        print(len(cumulative_price))
        print(len(capacity))

        plt.title( "Tickets sold over time ")
        plt.xlabel("Time")
        plt.ylabel("Price / Capacity")
        plt.plot(total_price, label = "Price per ticket sold")
        #plt.plot(cumulative_price, label = "Cumulative price")
        plt.plot(capacity, label = "Capacity")
        plt.legend()
        plt.show()

        plt.title( "Tickets sold over time ")
        plt.xlabel("Time")
        plt.ylabel("Price")
        #plt.plot(total_price, label = "Price per ticket sold")
        plt.plot(cumulative_price, label = "Cumulative price")
        #plt.plot(capacity, label = "Capacity")
        plt.legend()
        plt.show()

homework_one("C")



