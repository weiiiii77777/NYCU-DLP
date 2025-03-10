import matplotlib.pyplot as plt

idx = []
for i in range(1, 51) : idx.append(i)

vgg_train_1   = [ 3.42, 11.92, 22.82, 33.44, 41.88, 48.60, 54.18, 59.19, 62.99, 65.93,    # data preprocessing
                  69.13, 71.16, 72.92, 75.09, 76.10, 77.89, 79.16, 79.45, 80.37, 81.33,
                  82.15, 83.13, 83.15, 84.23, 84.50, 84.87, 84.77, 85.49, 85.43, 85.98,
                  86.71, 86.99, 86.98, 87.39, 87.43, 88.49, 88.37, 88.47, 88.42, 88.66,
                  88.38, 89.23, 89.55, 89.45, 90.01, 89.72, 89.11, 89.99, 89.95, 90.66  ]  

vgg_valid_1   =  [ 6.8, 18.4, 25.2, 44.4, 46.4, 56.8, 65.4, 66.8, 58.6, 71.6,
                  69.0, 76.8, 78.6, 75.2, 81.2, 82.4, 83.4, 77.8, 82.0, 87.6,
                  84.6, 84.4, 85.8, 88.6, 86.0, 85.8, 88.8, 88.2, 90.2, 89.4,
                  88.6, 88.6, 87.0, 89.4, 89.6, 86.4, 89.0, 92.0, 91.0, 90.0,
                  90.4, 88.6, 89.8, 90.6, 89.2, 93.4, 89.4, 87.6, 92.4, 90.8  ]

resnet_train_1 = [ 6.59, 18.83, 29.63, 38.22, 44.77, 50.83, 56.26, 60.46, 63.85, 66.79, 
                  68.68, 71.22, 73.83, 74.90, 76.40, 78.38, 78.89, 79.42, 80.57, 82.17, 
                  83.01, 83.28, 83.46, 84.61, 84.62, 85.22, 85.32, 85.86, 86.49, 86.37,
                  87.53, 87.61, 87.34, 87.43, 87.72, 88.20, 88.06, 88.28, 88.65, 88.30,
                  89.02, 89.25, 89.65, 89.30, 88.74, 89.34, 89.30, 89.84, 88.80, 90.00  ]

resnet_valid_1 = [ 19.4, 33.6, 39.0, 48.0, 55.8, 61.6, 64.4, 64.2, 66.4, 73.2,
                   72.6, 73.8, 75.8, 74.2, 78.8, 82.6, 82.2, 72.8, 79.8, 81.4,
                   83.4, 84.4, 85.2, 80.8, 82.8, 85.6, 81.6, 85.4, 78.0, 81.2,
                   85.4, 85.8, 89.0, 88.0, 82.4, 80.6, 84.8, 87.8, 85.2, 83.6,
                   87.2, 87.2, 88.2, 82.8, 87.4, 85.4, 85.6, 84.2, 86.6, 84.6  ]

vgg_train_2  = [    6.13, 18.65, 32.55, 44.58, 52.22, 59.77, 65.06, 69.40, 72.84, 75.55,      # lr = 0.02
                   78.77, 80.09, 81.93, 83.04, 85.55, 86.09, 86.84, 87.53, 88.69, 88.83, 
                   90.00, 89.65, 90.23, 90.88, 91.04, 92.59, 91.64, 91.79, 92.48, 93.28, 
                   92.96, 92.46, 93.44, 94.04, 93.43, 93.78, 93.83, 92.37, 94.64, 94.72,
                   94.53, 95.37, 93.98, 94.67, 95.50, 95.22, 93.93, 94.92, 94.30, 95.52 ]

vgg_valid_2  = [    8.0, 35.4, 44.6, 56.0, 58.2, 61.2, 73.6, 76.8, 77.2, 78.4, 
                   79.0, 79.2, 79.4, 83.2, 82.4, 81.2, 83.2, 77.4, 83.8, 79.6,
                   85.0, 83.8, 83.0, 82.2, 81.6, 81.6, 82.0, 83.4, 86.6, 82.2,
                   83.4, 84.8, 85.4, 81.6, 79.0, 85.8, 78.4, 83.8, 85.2, 86.4, 
                   85.4, 83.8, 83.6, 85.6, 84.4, 82.2, 84.6, 84.4, 84.8, 86.8 ]

resnet_train_2 = [ 5.68, 20.44, 34.91, 47.24, 56.26, 62.35, 67.55, 71.61, 74.37, 78.12,
                  80.27, 83.80, 85.54, 86.64, 88.71, 90.77, 90.87, 91.60, 91.86, 93.27, 
                  93.50, 94.57, 94.93, 93.22, 94.07, 94.02, 92.42, 96.81, 94.72, 95.24,
                  95.34, 93.21, 95.24, 94.73, 93.87, 95.07, 95.22, 92.08, 93.77, 96.26,
                  95.62, 94.77, 93.42, 93.91, 93.83, 95.81, 95.91, 95.92, 93.23, 94.89  ]

resnet_valid_2 = [ 15.4, 38.2, 51.4, 57.2, 63.2, 65.6, 68.2, 73.6, 75.8, 73.8, 
                   75.0, 80.6, 73.2, 72.4, 78.8, 75.6, 79.4, 80.8, 79.6, 81.0, 
                   79.0, 77.4, 80.2, 77.0, 76.6, 72.8, 78.8, 79.2, 81.2, 78.0,
                   81.0, 79.0, 79.2, 68.0, 77.0, 83.4, 74.8, 80.6, 82.4, 82.0,
                   74.8, 79.0, 68.8, 67.2, 78.6, 78.6, 77.0, 81.0, 75.6, 75.2  ]

vgg_train_3  = [   5.49, 20.14, 38.39, 51.22, 61.06, 67.46, 72.32, 76.32, 79.06, 81.38,     # lr = 0.0008
                  84.46, 85.72, 87.24, 88.85, 90.43, 91.27, 92.01, 92.45, 93.72, 94.49,
                  95.73, 95.74, 96.09, 95.41, 97.09, 96.07, 97.46, 98.08, 97.40, 96.26,
                  96.45, 97.82, 98.82, 98.37, 98.41, 98.91, 98.82, 97.96, 98.44, 97.62,
                  98.13, 99.03, 97.97, 98.62, 97.66, 99.06, 97.01, 99.06, 96.98, 99.18  ]

vgg_valid_3  = [   16.2, 37.8, 52.6, 62.8, 72.8, 73.2, 77.4, 74.8, 79.0, 83.2,
                   85.8, 86.2, 85.4, 81.2, 87.3, 75.4, 89.4, 86.0, 85.4, 86.6,
                   88.2, 90.2, 84.8, 89.2, 90.4, 86.0, 89.2, 89.4, 88.0, 87.2,
                   90.2, 89.2, 91.2, 90.4, 89.2, 90.0, 88.6, 89.4, 88.0, 87.2,
                   89.4, 88.8, 89.2, 89.2, 90.4, 88.2, 89.0, 90.0, 86.4, 92.2  ]

resnet_train_3 = [ 8.26, 25.08, 39.61, 50.56, 57.71, 64.65, 70.17, 74.65, 78.32, 82.13,
                  85.88, 87.52, 91.43, 93.66, 93.04, 95.82, 97.69, 98.18, 98.25, 97.80,
                  97.86, 98.36, 99.75, 99.13, 99.88, 99.62, 99.14, 98.41, 95.75, 97.58,
                  98.65, 99.55, 98.98, 99.92, 99.81, 98.66, 99.76, 98.98, 94.10, 95.70,
                  98.11, 99.61, 99.98, 99.90, 99.63, 99.91, 97.02, 93.81, 95.78, 98.72  ]

resnet_valid_3 = [ 20.4, 38.0, 54.4, 61.8, 65.8, 68.8, 73.4, 73.6, 69.0, 78.0,
                   79.8, 78.2, 83.4, 80.0, 82.0, 81.0, 83.4, 83.2, 83.6, 85.6,
                   80.8, 85.4, 84.8, 86.0, 87.0, 85.2, 81.6, 77.6, 76.8, 78.8,
                   84.8, 87.8, 85.4, 87.4, 85.8, 84.2, 86.4, 84.2, 75.0, 77.8,
                   84.4, 86.4, 86.6, 89.4, 85.0, 87.2, 74.2, 76.0, 81.0, 86.8,  ]

vgg_train_4 = [2.82 ,   5.78 ,  10.51 ,  17.07 ,  24.25 ,  29.51 ,  34.76 ,  39.82 ,  44.82 ,  48.09 ,  # adagrad optimizer
              51.29 ,  54.22 ,  56.55 ,  59.66 ,  61.72 ,  64.20 ,  65.47 ,  66.48 ,  69.49 ,  70.45 ,  
              71.32 ,  72.62 ,  73.76 ,  75.29 ,  76.36 ,  76.19 ,  77.24 ,  78.10 ,  78.81 ,  79.66 ,  
              80.52 ,  80.59 ,  81.42 ,  82.03 ,  82.35 ,  82.79 ,  83.04 ,  84.05 ,  84.30 ,  84.88 ,  
              85.62 ,  85.24 ,  86.35 ,  86.65 ,  86.68 ,  87.02 ,  87.41 ,  88.26 ,  88.41 ,  88.45 ]

vgg_valid_4 = [4.2 ,  7.4 ,  14.6 ,  19.4 ,  30.4 ,  31.4 ,  24.6 ,  45.8 ,  47.0 ,  57.6 ,  
              52.2 ,  62.4 ,  67.2 ,  70.4 ,  60.6 ,  62.6 ,  53.0 ,  71.2 ,  71.6 ,  73.4 ,  
              76.6 ,  62.6 ,  79.4 ,  74.4 ,  79.4 ,  81.0 ,  80.4 ,  82.2 ,  80.0 ,  82.4 ,  
              83.0 ,  81.4 ,  74.8 ,  85.4 ,  83.0 ,  87.2 ,  82.4 ,  86.6 ,  87.6 ,  87.4 ,  
              86.2 ,  87.8 ,  86.6 ,  76.6 ,  84.4 ,  84.0 ,  89.4 ,  81.6 ,  89.2 ,  87.4 ]

resnet_train_4 = [4.92 ,  17.36 ,  30.78 ,  40.24 ,  47.74 ,  53.75 ,  57.78 ,  61.66 ,  63.60 ,  66.41 ,  
                 69.68 ,  71.76 ,  73.64 ,  74.77 ,  76.20 ,  77.78 ,  79.22 ,  80.43 ,  81.46 ,  82.64 ,  
                 83.28 ,  84.21 ,  84.64 ,  85.44 ,  86.19 ,  86.66 ,  86.87 ,  87.68 ,  88.92 ,  89.17 ,  
                 89.53 ,  89.73 ,  90.14 ,  90.54 ,  91.27 ,  91.81 ,  91.93 ,  91.80 ,  91.98 ,  92.96 ,  
                 93.13 ,  93.24 ,  93.61 ,  93.80 ,  93.64 ,  94.53 ,  94.62 ,  94.74 ,  94.72 ,  95.30 ]

resnet_valid_4 = [11.4 ,  27.0 ,  41.4 ,  47.8 ,  52.2 ,  56.4 ,  57.4 ,  66.4 ,  68.0 ,  73.4 ,  
                  75.2 ,  74.0 ,  78.0 ,  73.6 ,  74.4 ,  79.6 ,  80.4 ,  82.6 ,  81.6 ,  82.6 ,  
                  83.4 ,  83.2 ,  82.6 ,  76.4 ,  83.0 ,  84.6 ,  86.8 ,  87.4 ,  86.8 ,  85.6 ,  
                  85.6 ,  87.2 ,  85.6 ,  85.4 ,  86.8 ,  86.4 ,  87.4 ,  87.2 ,  87.6 ,  88.8 ,  
                  86.2 ,  86.6 ,  86.6 ,  87.4 ,  87.8 ,  86.8 ,  85.6 ,  88.0 ,  85.0 ,  85.4 ]

resnet_train_5 = [2.83,  6.80, 10.00, 14.75, 20.79, 27.1, 31.94, 38.22, 42.93, 45.47,       # adam optimizer
                 48.63, 51.79, 54.92, 57.76, 59.34, 61.61, 62.85, 65.49, 67.02, 68.11, 
                 69.53, 71.38, 72.44, 72.74, 74.34, 75.82, 76.33, 77.62, 78.86, 79.01, 
                 80.52, 80.78, 81.72, 82.04, 82.90, 83.75, 83.05, 84.29, 85.05, 84.90,
                 85.61, 86.57, 86.90, 87.36, 85.56, 88.69, 88.66, 88.69, 89.38, 89.60 ]

resnet_valid_5 = [4.4,  8.8, 15.8, 11.2, 28.2, 34.0, 41.0, 44.0, 47.2, 46.0,
                 59.0, 55.4, 58.4, 64.4, 62.6, 66.4, 71.0, 70.8, 72.2, 74.4, 
                 76.4, 73.6, 76.8, 76.8, 78.0, 79.4, 81.0, 82.2, 78.4, 78.6,
                 83.4, 81.8, 82.8, 80.2, 80.4, 84.2, 82.2, 81.4, 85.2, 86.4,
                 85.9, 85.0, 86.4, 84.4, 85.8, 85.2, 86.2, 86.0, 86.4, 85.4 ]

vgg_train_6 = [6.63, 23.92, 41.07, 53.24, 60.70, 66.98, 71.96, 75.17, 78.83, 80.71,         # without data preprocessing
              82.70, 84.46, 86.64, 87.96, 89.07, 90.30, 91.15, 92.16, 92.46, 93.4,
              93.41, 94.60, 94.74, 95.29, 95.20, 95.62, 95.93, 95.73, 95.96, 96.83,
              96.97, 97.46, 95.75, 97.32, 97.69, 97.63, 97.55, 95.56, 98.24, 97.20,
              97.59, 96.62, 98.36, 98.78, 96.01, 98.26, 96.40, 95.56, 98.69, 99.45 ]

vgg_valid_6 = [16.8, 37.6, 57.2, 65.0, 69.0, 73.0, 78.2, 79.6, 80.2, 82.2, 
               82.2, 84.8, 85.4, 84.8, 84.6, 88.4, 84.6, 86.0, 84.0, 85.6,
               86.4, 85.8, 89.2, 85.6, 84.0, 88.0, 86.2, 85.6, 90.0, 87.8,
               85.2, 87.2, 90.2, 88.4, 85.6, 88.4, 83.6, 85.8, 89.4, 85.2,
               86.2, 88.6, 90.2, 88.2, 87.6, 84.2, 89.2, 87.6, 91.2, 89.8 ]

resnet_train_6 = [7.07, 22.05, 36.45, 48.49, 57.49, 64.99, 70.26, 74.97, 79.3, 83.36,
                 85.94, 89.10, 91.02, 93.59, 95.55, 96.79, 97.47, 96.69, 97.27, 98.00,
                 98.41, 99.06, 99.10, 99.29, 96.15, 97.72, 99.15, 98.73, 95.91, 95.46,
                 96.48, 98.88, 97.98, 99.20, 99.84, 99.94, 99.94, 99.87, 99.79, 92.18, 
                 90.34, 96.12, 98.37, 99.48, 99.79, 99.60, 99.66, 99.86, 99.99, 99.64 ]

resnet_valid_6 = [15.8, 35.8, 49.8, 60.2, 64.6, 70.8, 72.6, 73.2, 75.4, 79.6,
                  79.2, 76.2, 78.2, 83.2, 80.2, 82.0, 82.4, 82.0, 82.8, 83.2,
                  83.2, 85.0, 86.0, 85.8, 80.8, 82.0, 85.4, 80.2, 81.2, 82.2,
                  82.6, 84.0, 86.2, 85.8, 87.6, 89.2, 88.8, 89.8, 86.8, 68.8,
                  78.4, 83.8, 85.8, 90.4, 88.4, 88.0, 88.0, 89.8, 90.2, 69.2 ] 




if __name__ == "__main__" : 

    # path = 'output.txt'
    # f = open(path, 'w')
    # for i in range(50):
    #     print(vgg_train_1[i],', ', end=' ', file=f)
    #     if i % 10 == 9 :
    #         print(file=f)
    # f.close()
    
    plt.figure(figsize=(6,5))
    plt.plot(idx, vgg_train_1, label = 'VGG19_train_acc')    
    plt.plot(idx, vgg_valid_1, label = 'VGG19_valid_acc')
    plt.plot(idx, resnet_train_1, label = 'ResNet50_train_acc') 
    plt.plot(idx, resnet_valid_1, label = 'ResNet50_valid_acc') 
    plt.legend(['VGG19_train_acc', 'VGG19_valid_acc', 'ResNet50_train_acc', 'ResNet50_valid_acc'])
    # plt.plot(idx, resnet_train_5, label = 'ResNet50_train_acc') 
    # plt.plot(idx, resnet_valid_5, label = 'ResNet50_valid_acc') 
    # plt.legend(['ResNet50_train_acc', 'ResNet50_valid_acc'])
    plt.xlabel('Epoch')    
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve', loc='left')
    plt.title('with data preprocessing', loc='right')
    plt.savefig('./curve.jpg')