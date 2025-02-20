import matplotlib.pyplot as plt
import numpy as np

idx = []
for i in range(1, 301) : idx.append(i)

D = [0.125 , 0.125 , 0.11111 , 0.06944 , 0.08333 , 0.15278 , 0.11111 , 0.09722 , 0.16667 , 0.15278 , 
0.16667 , 0.11111 , 0.125 , 0.125 , 0.15278 , 0.15278 , 0.06944 , 0.08333 , 0.11111 , 0.125 , 
0.05556 , 0.19444 , 0.31944 , 0.15278 , 0.20833 , 0.19444 , 0.18056 , 0.29167 , 0.20833 , 0.19444 , 
0.23611 , 0.19444 , 0.25 , 0.22222 , 0.19444 , 0.29167 , 0.27778 , 0.31944 , 0.23611 , 0.34722 , 
0.16667 , 0.375 , 0.31944 , 0.31944 , 0.27778 , 0.34722 , 0.26389 , 0.30556 , 0.27778 , 0.34722 , 
0.27778 , 0.31944 , 0.31944 , 0.31944 , 0.29167 , 0.38889 , 0.27778 , 0.36111 , 0.36111 , 0.40278 , 
0.30556 , 0.36111 , 0.43056 , 0.33333 , 0.34722 , 0.44444 , 0.43056 , 0.33333 , 0.33333 , 0.43056 , 
0.44444 , 0.38889 , 0.375 , 0.38889 , 0.43056 , 0.47222 , 0.47222 , 0.43056 , 0.43056 , 0.43056 , 
0.47222 , 0.44444 , 0.38889 , 0.36111 , 0.38889 , 0.40278 , 0.47222 , 0.41667 , 0.41667 , 0.41667 , 
0.41667 , 0.41667 , 0.34722 , 0.40278 , 0.33333 , 0.41667 , 0.40278 , 0.44444 , 0.44444 , 0.47222 , 
0.41667 , 0.43056 , 0.41667 , 0.41667 , 0.5 , 0.40278 , 0.44444 , 0.41667 , 0.44444 , 0.5 , 
0.43056 , 0.43056 , 0.45833 , 0.41667 , 0.56944 , 0.47222 , 0.48611 , 0.54167 , 0.45833 , 0.48611 , 
0.51389 , 0.44444 , 0.41667 , 0.44444 , 0.48611 , 0.52778 , 0.56944 , 0.48611 , 0.44444 , 0.47222 , 
0.47222 , 0.375 , 0.45833 , 0.55556 , 0.45833 , 0.48611 , 0.44444 , 0.54167 , 0.52778 , 0.51389 , 
0.44444 , 0.44444 , 0.43056 , 0.59722 , 0.55556 , 0.55556 , 0.5 , 0.48611 , 0.45833 , 0.51389 , 
0.51389 , 0.52778 , 0.38889 , 0.45833 , 0.52778 , 0.38889 , 0.47222 , 0.52778 , 0.5 , 0.51389 , 
0.40278 , 0.48611 , 0.38889 , 0.48611 , 0.51389 , 0.55556 , 0.47222 , 0.5 , 0.58333 , 0.51389 , 
0.5 , 0.44444 , 0.54167 , 0.56944 , 0.44444 , 0.51389 , 0.43056 , 0.47222 , 0.48611 , 0.44444 , 
0.54167 , 0.55556 , 0.47222 , 0.52778 , 0.48611 , 0.47222 , 0.44444 , 0.45833 , 0.45833 , 0.52778 , 
0.48611 , 0.47222 , 0.47222 , 0.43056 , 0.43056 , 0.45833 , 0.44444 , 0.55556 , 0.43056 , 0.40278 , 
0.41667 , 0.51389 , 0.44444 , 0.52778 , 0.41667 , 0.5 , 0.55556 , 0.44444 , 0.44444 , 0.52778 , 
0.5 , 0.55556 , 0.45833 , 0.59722 , 0.47222 , 0.55556 , 0.51389 , 0.5 , 0.54167 , 0.61111 , 
0.51389 , 0.5 , 0.44444 , 0.61111 , 0.47222 , 0.5 , 0.52778 , 0.58333 , 0.5 , 0.625 , 
0.5 , 0.47222 , 0.48611 , 0.54167 , 0.52778 , 0.41667 , 0.55556 , 0.51389 , 0.44444 , 0.45833 , 
0.58333 , 0.45833 , 0.54167 , 0.44444 , 0.45833 , 0.56944 , 0.55556 , 0.55556 , 0.51389 , 0.52778 , 
0.58333 , 0.40278 , 0.56944 , 0.58333 , 0.63889 , 0.44444 , 0.44444 , 0.51389 , 0.625 , 0.61111 , 
0.56944 , 0.61111 , 0.63889 , 0.5 , 0.56944 , 0.625 , 0.48611 , 0.55556 , 0.44444 , 0.58333 , 
0.55556 , 0.56944 , 0.48611 , 0.56944 , 0.55556 , 0.55556 , 0.58333 , 0.51389 , 0.56944 , 0.52778 , 
0.58333 , 0.52778 , 0.55556 , 0.54167 , 0.55556 , 0.54167 , 0.48611 , 0.61111 , 0.56944 , 0.41667 , 
0.59722 , 0.54167 , 0.51389 , 0.56944 , 0.47222 , 0.52778 , 0.5 , 0.51389 , 0.51389 , 0.54167]

A_1 = [0.13889 , 0.16667 , 0.09722 , 0.06944 , 0.11111 , 0.16667 , 0.125 , 0.125 , 0.11111 , 0.16667 , 
0.22222 , 0.19444 , 0.06944 , 0.20833 , 0.19444 , 0.19444 , 0.18056 , 0.20833 , 0.22222 , 0.20833 , 
0.18056 , 0.18056 , 0.20833 , 0.15278 , 0.19444 , 0.18056 , 0.30556 , 0.31944 , 0.18056 , 0.20833 , 
0.22222 , 0.23611 , 0.19444 , 0.23611 , 0.26389 , 0.23611 , 0.19444 , 0.19444 , 0.20833 , 0.25 , 
0.27778 , 0.16667 , 0.23611 , 0.20833 , 0.25 , 0.22222 , 0.22222 , 0.23611 , 0.19444 , 0.27778 , 
0.29167 , 0.22222 , 0.19444 , 0.26389 , 0.34722 , 0.23611 , 0.26389 , 0.29167 , 0.26389 , 0.16667 , 
0.125 , 0.18056 , 0.15278 , 0.22222 , 0.20833 , 0.22222 , 0.16667 , 0.23611 , 0.16667 , 0.11111 , 
0.11111 , 0.13889 , 0.08333 , 0.08333 , 0.18056 , 0.16667 , 0.19444 , 0.20833 , 0.15278 , 0.22222 , 
0.15278 , 0.18056 , 0.16667 , 0.19444 , 0.19444 , 0.13889 , 0.18056 , 0.19444 , 0.11111 , 0.25 , 
0.18056 , 0.16667 , 0.15278 , 0.16667 , 0.11111 , 0.19444 , 0.23611 , 0.26389 , 0.26389 , 0.16667 , 
0.25 , 0.18056 , 0.20833 , 0.18056 , 0.20833 , 0.27778 , 0.19444 , 0.18056 , 0.22222 , 0.19444 , 
0.20833 , 0.22222 , 0.15278 , 0.18056 , 0.18056 , 0.18056 , 0.11111 , 0.11111 , 0.11111 , 0.15278 , 
0.11111 , 0.09722 , 0.19444 , 0.19444 , 0.19444 , 0.19444 , 0.16667 , 0.125 , 0.09722 , 0.19444 , 
0.15278 , 0.20833 , 0.125 , 0.13889 , 0.15278 , 0.09722 , 0.06944 , 0.125 , 0.15278 , 0.13889 , 
0.13889 , 0.11111 , 0.09722 , 0.15278 , 0.11111 , 0.125 , 0.15278 , 0.11111 , 0.09722 , 0.09722 , 
0.11111 , 0.11111 , 0.11111 , 0.13889 , 0.125 , 0.08333 , 0.06944 , 0.11111 , 0.08333 , 0.15278 , 
0.16667 , 0.13889 , 0.18056 , 0.02778 , 0.11111 , 0.125 , 0.13889 , 0.23611 , 0.125 , 0.15278 , 
0.16667 , 0.13889 , 0.18056 , 0.125 , 0.09722 , 0.11111 , 0.13889 , 0.15278 , 0.125 , 0.16667 , 
0.09722 , 0.09722 , 0.08333 , 0.13889 , 0.16667 , 0.19444 , 0.19444 , 0.16667 , 0.15278 , 0.11111 , 
0.20833 , 0.13889 , 0.125 , 0.09722 , 0.125 , 0.09722 , 0.11111 , 0.125 , 0.19444 , 0.13889 , 
0.125 , 0.11111 , 0.08333 , 0.05556 , 0.11111 , 0.11111 , 0.08333 , 0.09722 , 0.13889 , 0.125 , 
0.11111 , 0.125 , 0.125 , 0.09722 , 0.11111 , 0.08333 , 0.05556 , 0.125 , 0.09722 , 0.13889 , 
0.11111 , 0.16667 , 0.13889 , 0.09722 , 0.06944 , 0.09722 , 0.05556 , 0.06944 , 0.06944 , 0.06944 , 
0.09722 , 0.08333 , 0.09722 , 0.09722 , 0.125 , 0.08333 , 0.04167 , 0.09722 , 0.15278 , 0.15278 , 
0.125 , 0.125 , 0.15278 , 0.15278 , 0.06944 , 0.11111 , 0.19444 , 0.11111 , 0.09722 , 0.08333 , 
0.09722 , 0.11111 , 0.11111 , 0.11111 , 0.11111 , 0.125 , 0.125 , 0.125 , 0.125 , 0.15278 , 
0.15278 , 0.09722 , 0.15278 , 0.09722 , 0.11111 , 0.08333 , 0.18056 , 0.16667 , 0.20833 , 0.125 , 
0.09722 , 0.16667 , 0.18056 , 0.15278 , 0.13889 , 0.15278 , 0.15278 , 0.11111 , 0.13889 , 0.11111 , 
0.13889 , 0.22222 , 0.08333 , 0.15278 , 0.09722 , 0.05556 , 0.09722 , 0.06944 , 0.08333 , 0.15278 , 
0.11111 , 0.15278 , 0.18056 , 0.20833 , 0.15278 , 0.09722 , 0.08333 , 0.06944 , 0.125 , 0.16667]

A_10 = [0.13889 , 0.11111 , 0.30556 , 0.29167 , 0.26389 , 0.26389 , 0.23611 , 0.27778 , 0.25 , 0.31944 , 
0.36111 , 0.36111 , 0.375 , 0.31944 , 0.40278 , 0.43056 , 0.33333 , 0.31944 , 0.40278 , 0.29167 , 
0.33333 , 0.29167 , 0.38889 , 0.33333 , 0.36111 , 0.375 , 0.41667 , 0.38889 , 0.375 , 0.36111 , 
0.375 , 0.375 , 0.41667 , 0.41667 , 0.36111 , 0.47222 , 0.47222 , 0.41667 , 0.47222 , 0.375 , 
0.38889 , 0.44444 , 0.34722 , 0.41667 , 0.40278 , 0.41667 , 0.40278 , 0.41667 , 0.41667 , 0.38889 , 
0.36111 , 0.375 , 0.38889 , 0.41667 , 0.31944 , 0.38889 , 0.375 , 0.38889 , 0.38889 , 0.375 , 
0.40278 , 0.38889 , 0.34722 , 0.36111 , 0.36111 , 0.41667 , 0.43056 , 0.45833 , 0.375 , 0.44444 , 
0.38889 , 0.47222 , 0.40278 , 0.41667 , 0.40278 , 0.375 , 0.375 , 0.41667 , 0.36111 , 0.375 , 
0.40278 , 0.38889 , 0.43056 , 0.41667 , 0.41667 , 0.41667 , 0.45833 , 0.44444 , 0.47222 , 0.40278 , 
0.41667 , 0.38889 , 0.41667 , 0.375 , 0.36111 , 0.44444 , 0.375 , 0.36111 , 0.41667 , 0.44444 , 
0.40278 , 0.375 , 0.40278 , 0.40278 , 0.45833 , 0.38889 , 0.44444 , 0.41667 , 0.41667 , 0.40278 , 
0.43056 , 0.38889 , 0.40278 , 0.43056 , 0.41667 , 0.44444 , 0.41667 , 0.44444 , 0.45833 , 0.44444 , 
0.43056 , 0.41667 , 0.40278 , 0.38889 , 0.38889 , 0.40278 , 0.38889 , 0.40278 , 0.41667 , 0.40278 , 
0.40278 , 0.45833 , 0.43056 , 0.40278 , 0.47222 , 0.41667 , 0.40278 , 0.43056 , 0.43056 , 0.41667 , 
0.45833 , 0.44444 , 0.52778 , 0.375 , 0.44444 , 0.43056 , 0.43056 , 0.40278 , 0.41667 , 0.38889 , 
0.40278 , 0.38889 , 0.44444 , 0.375 , 0.45833 , 0.375 , 0.375 , 0.41667 , 0.375 , 0.375 , 
0.41667 , 0.36111 , 0.48611 , 0.41667 , 0.43056 , 0.41667 , 0.38889 , 0.43056 , 0.41667 , 0.375 , 
0.375 , 0.34722 , 0.41667 , 0.38889 , 0.41667 , 0.41667 , 0.40278 , 0.41667 , 0.43056 , 0.375 , 
0.36111 , 0.41667 , 0.40278 , 0.43056 , 0.38889 , 0.36111 , 0.38889 , 0.40278 , 0.41667 , 0.40278 , 
0.375 , 0.375 , 0.38889 , 0.44444 , 0.40278 , 0.36111 , 0.38889 , 0.375 , 0.375 , 0.40278 , 
0.47222 , 0.41667 , 0.43056 , 0.43056 , 0.45833 , 0.44444 , 0.40278 , 0.41667 , 0.41667 , 0.43056 , 
0.45833 , 0.45833 , 0.44444 , 0.43056 , 0.47222 , 0.44444 , 0.43056 , 0.44444 , 0.40278 , 0.41667 , 
0.38889 , 0.375 , 0.33333 , 0.33333 , 0.34722 , 0.31944 , 0.34722 , 0.33333 , 0.31944 , 0.375 , 
0.41667 , 0.33333 , 0.375 , 0.45833 , 0.45833 , 0.45833 , 0.44444 , 0.45833 , 0.36111 , 0.40278 , 
0.40278 , 0.41667 , 0.40278 , 0.38889 , 0.375 , 0.40278 , 0.375 , 0.40278 , 0.375 , 0.40278 , 
0.38889 , 0.41667 , 0.41667 , 0.36111 , 0.40278 , 0.34722 , 0.34722 , 0.375 , 0.43056 , 0.43056 , 
0.45833 , 0.41667 , 0.41667 , 0.40278 , 0.41667 , 0.40278 , 0.41667 , 0.41667 , 0.43056 , 0.38889 , 
0.36111 , 0.36111 , 0.36111 , 0.34722 , 0.375 , 0.34722 , 0.33333 , 0.34722 , 0.34722 , 0.33333 , 
0.33333 , 0.33333 , 0.33333 , 0.34722 , 0.36111 , 0.33333 , 0.34722 , 0.375 , 0.38889 , 0.40278 , 
0.38889 , 0.375 , 0.44444 , 0.41667 , 0.41667 , 0.40278 , 0.40278 , 0.38889 , 0.36111 , 0.36111]

A_100_1 = [0.09722 , 0.22222 , 0.13889 , 0.125 , 0.11111 , 0.20833 , 0.27778 , 0.375 , 0.36111 , 0.27778 , 
0.30556 , 0.27778 , 0.31944 , 0.31944 , 0.40278 , 0.375 , 0.38889 , 0.38889 , 0.40278 , 0.43056 , 
0.375 , 0.375 , 0.375 , 0.5 , 0.45833 , 0.47222 , 0.45833 , 0.54167 , 0.52778 , 0.5 , 
0.52778 , 0.47222 , 0.43056 , 0.61111 , 0.51389 , 0.52778 , 0.55556 , 0.45833 , 0.54167 , 0.52778 , 
0.51389 , 0.45833 , 0.38889 , 0.38889 , 0.43056 , 0.5 , 0.47222 , 0.48611 , 0.45833 , 0.375 , 
0.48611 , 0.44444 , 0.51389 , 0.41667 , 0.44444 , 0.55556 , 0.52778 , 0.45833 , 0.47222 , 0.54167 , 
0.51389 , 0.52778 , 0.44444 , 0.43056 , 0.52778 , 0.47222 , 0.5 , 0.61111 , 0.51389 , 0.52778 , 
0.43056 , 0.52778 , 0.51389 , 0.5 , 0.51389 , 0.52778 , 0.625 , 0.55556 , 0.55556 , 0.61111 , 
0.58333 , 0.59722 , 0.58333 , 0.55556 , 0.55556 , 0.56944 , 0.63889 , 0.59722 , 0.51389 , 0.58333 , 
0.54167 , 0.52778 , 0.69444 , 0.65278 , 0.56944 , 0.56944 , 0.54167 , 0.52778 , 0.48611 , 0.58333 , 
0.61111 , 0.65278 , 0.66667 , 0.63889 , 0.59722 , 0.58333 , 0.68056 , 0.63889 , 0.625 , 0.59722 , 
0.59722 , 0.63889 , 0.59722 , 0.68056 , 0.61111 , 0.73611 , 0.63889 , 0.66667 , 0.625 , 0.69444 , 
0.69444 , 0.66667 , 0.73611 , 0.59722 , 0.69444 , 0.61111 , 0.625 , 0.65278 , 0.66667 , 0.73611 , 
0.70833 , 0.69444 , 0.69444 , 0.69444 , 0.70833 , 0.73611 , 0.65278 , 0.69444 , 0.65278 , 0.75 , 
0.66667 , 0.66667 , 0.68056 , 0.65278 , 0.65278 , 0.72222 , 0.68056 , 0.70833 , 0.70833 , 0.69444 , 
0.65278 , 0.65278 , 0.66667 , 0.58333 , 0.68056 , 0.63889 , 0.625 , 0.68056 , 0.69444 , 0.69444 , 
0.65278 , 0.61111 , 0.63889 , 0.63889 , 0.65278 , 0.68056 , 0.65278 , 0.61111 , 0.72222 , 0.66667 , 
0.70833 , 0.68056 , 0.69444 , 0.66667 , 0.70833 , 0.66667 , 0.72222 , 0.69444 , 0.66667 , 0.70833 , 
0.66667 , 0.68056 , 0.72222 , 0.69444 , 0.65278 , 0.63889 , 0.66667 , 0.66667 , 0.63889 , 0.63889 , 
0.66667 , 0.61111 , 0.65278 , 0.69444 , 0.68056 , 0.68056 , 0.68056 , 0.72222 , 0.68056 , 0.69444 , 
0.65278 , 0.65278 , 0.70833 , 0.70833 , 0.63889 , 0.65278 , 0.68056 , 0.69444 , 0.61111 , 0.61111 , 
0.68056 , 0.63889 , 0.68056 , 0.69444 , 0.65278 , 0.65278 , 0.66667 , 0.72222 , 0.65278 , 0.70833 , 
0.68056 , 0.69444 , 0.70833 , 0.66667 , 0.70833 , 0.65278 , 0.68056 , 0.75 , 0.68056 , 0.65278 , 
0.70833 , 0.68056 , 0.72222 , 0.70833 , 0.70833 , 0.70833 , 0.68056 , 0.66667 , 0.63889 , 0.69444 , 
0.66667 , 0.66667 , 0.70833 , 0.73611 , 0.66667 , 0.69444 , 0.73611 , 0.70833 , 0.72222 , 0.72222 , 
0.75 , 0.69444 , 0.72222 , 0.73611 , 0.72222 , 0.72222 , 0.77778 , 0.73611 , 0.75 , 0.73611 , 
0.69444 , 0.72222 , 0.70833 , 0.77778 , 0.73611 , 0.66667 , 0.68056 , 0.68056 , 0.73611 , 0.72222 , 
0.61111 , 0.65278 , 0.63889 , 0.72222 , 0.70833 , 0.69444 , 0.69444 , 0.70833 , 0.66667 , 0.73611 , 
0.70833 , 0.75 , 0.72222 , 0.69444 , 0.68056 , 0.73611 , 0.70833 , 0.72222 , 0.69444 , 0.66667 , 
0.68056 , 0.66667 , 0.73611 , 0.70833 , 0.70833 , 0.69444 , 0.69444 , 0.75 , 0.73611 , 0.72222]

A_100_2 = [0.19444 , 0.16667 , 0.22222 , 0.27778 , 0.29167 , 0.19444 , 0.20833 , 0.20833 , 0.375 , 0.34722 , 
0.36111 , 0.375 , 0.40278 , 0.5 , 0.45833 , 0.48611 , 0.51389 , 0.47222 , 0.52778 , 0.52778 , 
0.55556 , 0.52778 , 0.52778 , 0.61111 , 0.61111 , 0.59722 , 0.65278 , 0.59722 , 0.58333 , 0.55556 , 
0.58333 , 0.55556 , 0.59722 , 0.66667 , 0.59722 , 0.56944 , 0.61111 , 0.58333 , 0.625 , 0.625 , 
0.61111 , 0.66667 , 0.68056 , 0.63889 , 0.69444 , 0.65278 , 0.69444 , 0.70833 , 0.69444 , 0.70833 , 
0.75 , 0.63889 , 0.66667 , 0.65278 , 0.69444 , 0.72222 , 0.625 , 0.73611 , 0.68056 , 0.69444 , 
0.70833 , 0.65278 , 0.61111 , 0.69444 , 0.65278 , 0.69444 , 0.63889 , 0.68056 , 0.70833 , 0.72222 , 
0.69444 , 0.73611 , 0.73611 , 0.79167 , 0.68056 , 0.69444 , 0.72222 , 0.625 , 0.72222 , 0.69444 , 
0.66667 , 0.69444 , 0.72222 , 0.66667 , 0.75 , 0.625 , 0.69444 , 0.68056 , 0.72222 , 0.76389 , 
0.68056 , 0.73611 , 0.70833 , 0.72222 , 0.68056 , 0.68056 , 0.66667 , 0.68056 , 0.69444 , 0.63889 , 
0.65278 , 0.68056 , 0.73611 , 0.73611 , 0.66667 , 0.70833 , 0.72222 , 0.73611 , 0.68056 , 0.70833 , 
0.69444 , 0.69444 , 0.72222 , 0.65278 , 0.70833 , 0.69444 , 0.68056 , 0.69444 , 0.69444 , 0.70833 , 
0.69444 , 0.70833 , 0.69444 , 0.72222 , 0.66667 , 0.625 , 0.70833 , 0.68056 , 0.69444 , 0.65278 , 
0.65278 , 0.77778 , 0.75 , 0.75 , 0.73611 , 0.69444 , 0.77778 , 0.70833 , 0.69444 , 0.75 , 
0.70833 , 0.65278 , 0.68056 , 0.70833 , 0.69444 , 0.68056 , 0.66667 , 0.68056 , 0.70833 , 0.66667 , 
0.70833 , 0.68056 , 0.72222 , 0.66667 , 0.72222 , 0.73611 , 0.70833 , 0.63889 , 0.75 , 0.625 , 
0.66667 , 0.73611 , 0.77778 , 0.65278 , 0.72222 , 0.72222 , 0.66667 , 0.66667 , 0.625 , 0.75 , 
0.73611 , 0.75 , 0.72222 , 0.76389 , 0.73611 , 0.75 , 0.75 , 0.70833 , 0.65278 , 0.69444 , 
0.73611 , 0.77778 , 0.69444 , 0.73611 , 0.69444 , 0.69444 , 0.69444 , 0.70833 , 0.72222 , 0.69444 , 
0.76389 , 0.72222 , 0.76389 , 0.72222 , 0.68056 , 0.69444 , 0.69444 , 0.68056 , 0.69444 , 0.75 , 
0.65278 , 0.68056 , 0.72222 , 0.80556 , 0.80556 , 0.77778 , 0.75 , 0.68056 , 0.75 , 0.73611 , 
0.72222 , 0.75 , 0.79167 , 0.76389 , 0.77778 , 0.75 , 0.70833 , 0.72222 , 0.73611 , 0.76389 , 
0.72222 , 0.76389 , 0.75 , 0.77778 , 0.76389 , 0.73611 , 0.75 , 0.75 , 0.73611 , 0.79167 , 
0.73611 , 0.76389 , 0.73611 , 0.75 , 0.75 , 0.75 , 0.76389 , 0.72222 , 0.73611 , 0.72222 , 
0.76389 , 0.76389 , 0.75 , 0.73611 , 0.73611 , 0.75 , 0.76389 , 0.75 , 0.70833 , 0.75 , 
0.72222 , 0.72222 , 0.69444 , 0.76389 , 0.73611 , 0.73611 , 0.75 , 0.79167 , 0.73611 , 0.77778 , 
0.73611 , 0.76389 , 0.75 , 0.79167 , 0.75 , 0.73611 , 0.79167 , 0.75 , 0.73611 , 0.75 , 
0.77778 , 0.73611 , 0.76389 , 0.68056 , 0.72222 , 0.80556 , 0.79167 , 0.79167 , 0.81944 , 0.73611 , 
0.77778 , 0.75 , 0.75 , 0.75 , 0.76389 , 0.76389 , 0.77778 , 0.77778 , 0.73611 , 0.77778 , 
0.79167 , 0.79167 , 0.81944 , 0.76389 , 0.79167 , 0.79167 , 0.80556 , 0.80556 , 0.77778 , 0.70833]

A_100_4 = [0.20833 , 0.19444 , 0.19444 , 0.26389 , 0.38889 , 0.33333 , 0.5 , 0.44444 , 0.47222 , 0.47222 , 
0.43056 , 0.47222 , 0.52778 , 0.54167 , 0.52778 , 0.5 , 0.55556 , 0.55556 , 0.56944 , 0.52778 , 
0.56944 , 0.58333 , 0.59722 , 0.59722 , 0.61111 , 0.66667 , 0.58333 , 0.63889 , 0.625 , 0.68056 , 
0.69444 , 0.72222 , 0.66667 , 0.70833 , 0.70833 , 0.66667 , 0.625 , 0.66667 , 0.66667 , 0.66667 , 
0.66667 , 0.66667 , 0.63889 , 0.66667 , 0.69444 , 0.66667 , 0.66667 , 0.69444 , 0.68056 , 0.69444 , 
0.70833 , 0.66667 , 0.68056 , 0.68056 , 0.70833 , 0.69444 , 0.68056 , 0.69444 , 0.72222 , 0.72222 , 
0.77778 , 0.69444 , 0.69444 , 0.73611 , 0.70833 , 0.76389 , 0.76389 , 0.72222 , 0.75 , 0.70833 , 
0.68056 , 0.73611 , 0.70833 , 0.73611 , 0.70833 , 0.68056 , 0.73611 , 0.79167 , 0.72222 , 0.72222 , 
0.80556 , 0.76389 , 0.80556 , 0.77778 , 0.77778 , 0.75 , 0.77778 , 0.77778 , 0.75 , 0.77778 , 
0.79167 , 0.76389 , 0.72222 , 0.77778 , 0.76389 , 0.75 , 0.69444 , 0.75 , 0.75 , 0.72222 , 
0.77778 , 0.73611 , 0.79167 , 0.68056 , 0.75 , 0.76389 , 0.77778 , 0.75 , 0.76389 , 0.69444 , 
0.76389 , 0.79167 , 0.75 , 0.77778 , 0.77778 , 0.80556 , 0.76389 , 0.77778 , 0.73611 , 0.81944 , 
0.79167 , 0.76389 , 0.70833 , 0.77778 , 0.77778 , 0.80556 , 0.84722 , 0.77778 , 0.75 , 0.80556 , 
0.76389 , 0.75 , 0.76389 , 0.80556 , 0.75 , 0.73611 , 0.79167 , 0.79167 , 0.81944 , 0.73611 , 
0.76389 , 0.76389 , 0.77778 , 0.80556 , 0.76389 , 0.73611 , 0.79167 , 0.76389 , 0.76389 , 0.73611 , 
0.79167 , 0.90278 , 0.81944 , 0.80556 , 0.79167 , 0.79167 , 0.80556 , 0.80556 , 0.77778 , 0.79167 , 
0.81944 , 0.76389 , 0.69444 , 0.83333 , 0.77778 , 0.84722 , 0.83333 , 0.79167 , 0.81944 , 0.76389 , 
0.79167 , 0.81944 , 0.77778 , 0.76389 , 0.79167 , 0.81944 , 0.81944 , 0.80556 , 0.83333 , 0.80556 , 
0.79167 , 0.81944 , 0.77778 , 0.75 , 0.81944 , 0.83333 , 0.80556 , 0.79167 , 0.81944 , 0.79167 , 
0.81944 , 0.77778 , 0.81944 , 0.80556 , 0.80556 , 0.81944 , 0.81944 , 0.81944 , 0.80556 , 0.83333 , 
0.86111 , 0.79167 , 0.81944 , 0.80556 , 0.79167 , 0.81944 , 0.84722 , 0.79167 , 0.79167 , 0.84722 , 
0.76389 , 0.88889 , 0.76389 , 0.79167 , 0.81944 , 0.80556 , 0.79167 , 0.79167 , 0.77778 , 0.84722 , 
0.84722 , 0.79167 , 0.83333 , 0.83333 , 0.76389 , 0.76389 , 0.80556 , 0.79167 , 0.80556 , 0.79167 , 
0.83333 , 0.84722 , 0.75 , 0.77778 , 0.77778 , 0.80556 , 0.80556 , 0.81944 , 0.77778 , 0.84722 , 
0.80556 , 0.79167 , 0.77778 , 0.73611 , 0.79167 , 0.81944 , 0.80556 , 0.75 , 0.75 , 0.79167 , 
0.70833 , 0.75 , 0.75 , 0.73611 , 0.76389 , 0.76389 , 0.75 , 0.70833 , 0.77778 , 0.76389 , 
0.75 , 0.77778 , 0.77778 , 0.76389 , 0.76389 , 0.77778 , 0.79167 , 0.80556 , 0.80556 , 0.76389 , 
0.76389 , 0.80556 , 0.76389 , 0.77778 , 0.77778 , 0.79167 , 0.76389 , 0.80556 , 0.79167 , 0.76389 , 
0.77778 , 0.77778 , 0.77778 , 0.72222 , 0.69444 , 0.76389 , 0.69444 , 0.73611 , 0.84722 , 0.79167 , 
0.76389 , 0.80556 , 0.83333 , 0.81944 , 0.79167 , 0.77778 , 0.81944 , 0.79167 , 0.81944 , 0.80556]


if __name__ == "__main__" : 

    plt.figure(figsize=(6,5))
    plt.plot(idx, A_100_1, label = 'ms 1')    
    plt.plot(idx, A_100_2, label = 'ms 2')
    plt.plot(idx, A_100_4, label = 'ms 3', color = 'red')    
    # plt.plot(idx, fid_diff_T_s_7, label = 'ms 4', color = 'green')
    plt.legend(['G/D = 1', 'G/D = 2', 'G/D = 4'])
    plt.xlabel('epoch')    
    plt.ylabel('test.json Accuracy')
    plt.title('test.json Accuracy on different Proportion of G/D', loc='center')
    # plt.title('Optimizer = Adam', loc='right')
    plt.savefig('./curve.jpg')

    # plt.figure(figsize=(6,5))
    # # plt.plot(idx1, train_loss_wo_sheduler, label = 'no_t')    
    # # plt.plot(idx1, valid_loss_wo_sheduler, label = 'no_v')
    # plt.plot(idx1, train_loss_w_sheduler, label = 'yes_t')    
    # plt.plot(idx1, valid_loss_w_sheduler, label = 'yes_v')
    # plt.legend(['Train Loss', 'Valid Loss'])
    # plt.xlabel('Epoch')    
    # plt.ylabel('Loss')
    # plt.title('Transformer Loss Curve', loc='center')
    # # plt.title('Optimizer = Adam', loc='right')
    # plt.savefig('./curve.jpg')