# Patterns that CNN recognize

Note that the results here are visualized pattern tensors.

They are not exactly pattern images.

## Algorithm (using PyTorch)
```
  import torch
  import torch.autograd.Variable as Variable
  import torch.nn as nn
  import torch.optim as optim
  import torchvision.models as models
  import torchvision.transforms as transforms
  
  
  alexnet = models.alexnet(pretrained=True)
  alexnet = alexnet.cuda()
  
  for param in alexnet.parameters():
    param.requires_grad = False
    
  unloader = transforms.ToPILImage()
  
  for c in range(1000):
    label = Variable(torch.cuda.LongTensor([c]))

    input_img = torch.zeros((1, 3, 224, 224)).type(torch.cuda.FloatTensor)
    input_img = Variable(input_img, requires_grad = True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([input_img], lr=1.0)

    for epoch in range(1000):

        optimizer.zero_grad()

        output = alexnet(input_img)
        loss = criterion(output, label)
        
        if epoch == 999:
            print(loss)

        loss.backward(retain_variables=True)
        optimizer.step()
    
    pattern_tensor = unloader(input_img.data.clone().cpu().view(3, 224, 224))
    pattern_tensor.save('./alexnet/image_'+str(c)+'.jpg')
```

## AlexNet
### Class 0 tench, Tinca tinca
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_0.jpg)
Loss = 1.00000e-04 * 4.4632

### Class 1 goldfish, Carassius auratus
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_1.jpg)
Loss = 1.00000e-04 * 2.6703

### Class 2 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_2.jpg)
Loss = 1.00000e-03 * 2.8629

### Class 3 tiger shark, Galeocerdo cuvieri
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_3.jpg)
Loss = 1.00000e-04 * 7.9918

### Class 4 hammerhead, hammerhead shark
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_4.jpg)
Loss = 1.00000e-04 * 1.8120

### Class 5 electric ray, crampfish, numbfish, torpedo
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_5.jpg)
Loss = 1.00000e-04 * 3.1281

### Class 6 stingray
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_6.jpg)
Loss = 1.00000e-05 * 5.1498

### Class 7 cock
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_7.jpg)
Loss = 1.00000e-03 * 2.1839

### Class 8 hen
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_8.jpg)
Loss = 1.00000e-03 * 2.1420

### Class 9 ostrich, Struthio camelus
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_9.jpg)
Loss = 1.00000e-05 * 5.9128

### Class 10 brambling, Fringilla montifringilla
![image](https://github.com/YinTaiChen/Patterns-that-CNN-buy/blob/master/alexnet_0_10/image_10.jpg)
Loss = 1.00000e-04 * 6.6757
