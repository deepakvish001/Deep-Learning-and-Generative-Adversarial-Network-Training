import numpy as np

data = [50, 55, 70, 80, 130, 150, 155, 160]
labels = [1, 1, 1, 1, -1, -1, -1, -1]


trial = 0
while(1):
    
    try:
        error = 0
        # random initilization
        weight = np.random.rand(2) * 2 - 1
        # looping over data elements
        for i in range(len(data)):
            # checking if the predicition is right
            if np.sign(weight[0] * data[i] + weight[1]) != labels[i]:
                error += 1
        if error == 0:
            break
        else:
            trial += 1
    except:
        print("Number of trials:", trial)
        exit()

print("Number of trials:", trial)


















# trial = 0
# error = 1
# while(error):
#     try:
#         weight = np.random.rand(2) * 2 - 1
#         error = np.sum(np.sign(weight[0] * np.array(data) + weight[1]) - np.array(labels))
#         trial += 1
#     except:
#         print(trial)
#         exit()

# print("Number of trials:", trial)
