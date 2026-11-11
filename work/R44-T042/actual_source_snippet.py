# A 4x5 array: 14 border ones and 6 interior zeros
Z_4x5 = np.ones((4, 5))
Z_4x5[1:-1, 1:-1] = 0
print(Z_4x5)

# A 1x3 array: no interior, so all elements remain 1
Z_1x3 = np.ones((1, 3))
Z_1x3[1:-1, 1:-1] = 0
print(Z_1x3)
